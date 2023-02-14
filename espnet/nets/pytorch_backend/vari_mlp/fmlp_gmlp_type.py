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


class FourierGatingUnit(nn.Module):
    def __init__(self, eunits, act = nn.Identity(), kernel=15):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(eunits//2, kernel, dtype=torch.float32)*0.02)

        self.norm = nn.LayerNorm(eunits//2)
        self.act = act

    def forward(self, x, tiny_attn=None):

        res, gate = x.chunk(2, dim = -1)

        gate = self.norm(gate)
        n = gate.shape[1]

        gate = gate.permute(0,2,1)
        gate = torch.fft.rfft(gate, dim=-1, n=n)
        weight = torch.fft.rfft(self.weight, dim=-1, n=n)

        real = gate.real*weight.real + gate.imag*weight.imag
        imag = gate.imag*weight.real - gate.real*weight.imag
        gate = torch.view_as_complex(torch.cat([real[..., None], imag[..., None]], dim=-1))

        gate = torch.fft.irfft(gate, n=n, dim=-1).permute(0,2,1)
        gate = gate[:, :n, :]

        if tiny_attn is not None:
            gate = gate + tiny_attn

        return self.act(gate) * res


class FMLPBlock2(nn.Module):
    def __init__(
        self,
        adim,
        eunits,
        act=nn.Identity(),
        act_in=nn.GELU(),
        act_out=nn.Identity(),
        attn_dim=0,
        causal=False,
        kernel=15,
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

        self.fgu = FourierGatingUnit(eunits, act, kernel) 
        self.attn = Attention(adim, eunits // 2, attn_dim, causal) if attn_dim>0 else None

    def forward(self, x):
        residual = x
        
        x = self.pre_norm(x)

        tiny_attn = self.attn(x) if self.attn is not None else None
    
        x = self.proj_in(x)
        x = self.act_in(x)
        
        x = self.fgu(x, tiny_attn=tiny_attn)

        x = self.proj_out(x)
        x = self.act_out(x)

        return residual + x


class  FMLPEncoder2(nn.Module):
    def __init__(
        self,
        elayers,
        adim,
        eunits,
        attn_dim = 0,
        causal = False,
        act=nn.Identity(),
        act_in=nn.GELU(),
        act_out=nn.Identity(),
        kernel=15,
        dropout=0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                FMLPBlock2(
                    adim = adim,
                    eunits = eunits,
                    attn_dim = attn_dim,
                    causal = causal,
                    act = act,
                    act_in =  act_in,
                    act_out = act_out,
                    kernel = kernel,
                    dropout=dropout,
                ) for i in range(elayers)
            ]
        )
        self.norm = nn.LayerNorm(adim)

    def forward(self, x):
        out = nn.Sequential(*self.layers)(x)
        out = self.norm(out)
        return out
