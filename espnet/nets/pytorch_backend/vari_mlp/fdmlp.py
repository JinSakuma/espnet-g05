import torch
import torch.nn.functional as F
from torch import nn, einsum


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class ComplexLinear(nn.Module):
    def __init__(self, num_input, num_output, init_eps=1e-3, initialize_bias=None, **kwargs):
        super().__init__()
        self.real = nn.Linear(num_input, num_output, **kwargs)
        self.imag = nn.Linear(num_input, num_output, **kwargs)

        if initialize_bias:
            nn.init.uniform_(self.real.weight, -init_eps, init_eps)
            nn.init.uniform_(self.imag.weight, -init_eps, init_eps)
            nn.init.constant_(self.real.bias, 1.)
            nn.init.constant_(self.imag.bias, 1.)

    def forward(self, x):

        if x.dtype in [torch.complex64, torch.complex128]:
            real = self.real(x.real) - self.imag(x.imag)
            imag = self.real(x.imag) + self.imag(x.real)
        else:
            real = self.real(x)
            imag = self.imag(x)

        out = torch.view_as_complex(
            torch.cat([real[..., None], imag[..., None]], dim=-1)
        )
        return out


class ComplexFeedForward(nn.Module):
    def __init__(self, dim, initialize_bias = True, init_eps = 1e-3, bias=True):
        super().__init__()
        self.fftdim = 2 * dim- 1
        self.proj = ComplexLinear(dim, dim, init_eps=init_eps/dim, initialize_bias=initialize_bias)

    def forward(self, x):
        n = x.shape[1]
        x = x.permute(0,2,1)
        x = torch.fft.rfft(x, dim=-1, n=self.fftdim)
        x = self.proj(x)
        x = torch.fft.irfft(x, dim=-1, n=self.fftdim)
        x = x[:,:,:n]
        x = x.permute(0,2,1)
        return x


class ComplexFeedForward2(nn.Module):
    def __init__(self, cdim, sdim, init_eps = 1e-3):
        super().__init__()
        self.fftdim = 2 * sdim - 1
        self.proj_channel = ComplexLinear(cdim, cdim, init_eps=init_eps/cdim, initialize_bias=True)
        self.proj_spatial = ComplexLinear(sdim, sdim, init_eps=init_eps/sdim, initialize_bias=True)

    def forward(self, x):
        n = x.shape[1]
        x = torch.fft.rfft(x, dim=-2, n=self.fftdim)
        x = self.proj_channel(x)
        x = x.permute(0,2,1)
        x = self.proj_spatial(x)
        x = x.permute(0,2,1)
        x = torch.fft.irfft(x, dim=-2, n=self.fftdim)
        x = x[:,:n,:]
        return x


class FourierDomainGatingUnit(nn.Module):
    def __init__(self, eunits, dim_fft, act = nn.Identity(), initialize_bias = True, init_eps = 1e-3):
        super().__init__()
        dim_out = eunits // 2

        self.complex_linear = ComplexFeedForward(dim_fft, initialize_bias=initialize_bias)
        self.norm = nn.LayerNorm(dim_out)
        self.act = act

    def forward(self, x, tiny_attn = None):
        res, gate = x.chunk(2, dim = -1)

        gate = self.norm(gate)
        gate = self.complex_linear(gate)

        if tiny_attn is not None:
            gate = gate + tiny_attn
        
        return self.act(gate) * res


class FourierDomainGatingUnit2(nn.Module):
    def __init__(self, eunits, dim_fft, act = nn.Identity(), initialize_bias=True, init_eps = 1e-3):
        super().__init__()
        dim_out = eunits // 2

        self.complex_linear = ComplexFeedForward2(dim_out, dim_fft)
        self.norm = nn.LayerNorm(dim_out)
        self.act = act

    def forward(self, x, tiny_attn = None):
        res, gate = x.chunk(2, dim = -1)

        gate = self.norm(gate)
        gate = self.complex_linear(gate)

        if tiny_attn is not None:
            gate = gate + tiny_attn

        return self.act(gate) * res


class FDMLPBlock(nn.Module):
    def __init__(
        self,
        adim,
        eunits,
        dim_fft=512,
        act=nn.Identity(),
        act_in=nn.GELU(),
        act_out=nn.Identity(),
        attn_dim=0,
        causal=False,
        fdmlp_type=1,
        dropout=0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(adim, eunits)
        self.act_in = act_in
        self.act_out = act_out
        self.proj_out = nn.Linear(eunits // 2, adim)
        self.pre_norm = nn.LayerNorm(adim)

        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        if fdmlp_type==1:
            self.tsgu = FourierDomainGatingUnit(eunits, dim_fft, act) 
        elif fdmlp_type==2:
            self.tsgu = FourierDomainGatingUnit2(eunits, dim_fft, act) 
        else:
            raise NotImplementedError

        self.attn = Attention(adim, eunits // 2, attn_dim, causal) if attn_dim>0 else None

    def forward(self, x):
        residual = x
        
        x = self.pre_norm(x)

        tiny_attn = self.attn(x) if self.attn is not None else None
    
        x = self.proj_in(x)
        x = self.act_in(x)
        
        x = self.tsgu(x, tiny_attn=tiny_attn)

        x = self.proj_out(x)
        x = self.act_out(x)

        return residual + x


class  FDMLPEncoder(nn.Module):
    def __init__(
        self,
        elayers,
        adim,
        eunits,
        dim_fft=512,
        attn_dim = 0,
        causal = False,
        act=nn.Identity(),
        act_in=nn.GELU(),
        act_out=nn.Identity(),
        fdmlp_type=1,
        dropout=0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                FDMLPBlock(
                    adim = adim,
                    eunits = eunits,
                    dim_fft = dim_fft,
                    attn_dim = attn_dim,
                    causal = causal,
                    act = act,
                    act_in =  act_in,
                    act_out = act_out,
                    fdmlp_type = fdmlp_type,
                    dropout=dropout,
                ) for i in range(elayers)
            ]
        )
        self.norm = nn.LayerNorm(adim)

    def forward(self, x):
        out = nn.Sequential(*self.layers)(x)
        out = self.norm(out)
        return out
