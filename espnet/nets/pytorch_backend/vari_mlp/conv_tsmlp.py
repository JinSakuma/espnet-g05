import torch
import torch.nn.functional as F
from torch import nn, einsum
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule


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


class TemporalShiftGatingUnit(nn.Module):
    def __init__(self, eunits, act = nn.Identity(), shift_size=1):
        super().__init__()
        
        dim_out = eunits // 2
        self.norm = nn.LayerNorm(dim_out)
        self.act = act
        self.shift_size = shift_size

    def forward(self, x, tiny_attn=None):
        res, gate = x.chunk(2, dim = -1) 

        gate = self.norm(gate)
        
        # temporal shift
        b, t, d = gate.size()
        pad = torch.zeros([b, self.shift_size, d//2]).to(gate.device)
        g1 = torch.cat([pad, gate[:, :, :d//2]], dim=1)
        g2 = torch.cat([gate[:, :, d//2:], pad], dim=1)
        gate = torch.cat([g1[:, :-self.shift_size, :], g2[:, self.shift_size:, :]], dim=2)

        if tiny_attn is not None:
            gate = gate + tiny_attn
        
        return self.act(gate) * res


class ConvTSMLPBlock(nn.Module):
    def __init__(
        self,
        adim,
        eunits,
        act=nn.Identity(),
        act_in=nn.GELU(),
        act_out=nn.Identity(),
        attn_dim=0,
        causal=False,
        shift_size=1,
        kernel=15,
        dropout=0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(adim, eunits)
        self.act_in = act_in
        self.act_out = act_out
        self.proj_out = nn.Linear(eunits // 2, adim)
        self.pre_norm = nn.LayerNorm(adim)
        self.mid_norm = nn.LayerNorm(eunits // 2)
        self.conv_module = ConvolutionModule(eunits//2, kernel, nn.GELU())

        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.tsgu = TemporalShiftGatingUnit(eunits, act, shift_size) 
        self.attn = Attention(adim, eunits // 2, attn_dim, causal) if attn_dim>0 else None

    def forward(self, x):
        residual = x
        
        x = self.pre_norm(x)

        tiny_attn = self.attn(x) if self.attn is not None else None
    
        x = self.proj_in(x)
        x = self.act_in(x)
        
        x = self.tsgu(x, tiny_attn=tiny_attn)

        x = self.mid_norm(x)
        x = self.conv_module(x)

        x = self.proj_out(x)
        x = self.act_out(x)

        return residual + x


class  TSMLPEncoder(nn.Module):
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
        shift_size=1,
        kernel=15,
        dropout=0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                TSMLPBlock(
                    adim = adim,
                    eunits = eunits,
                    attn_dim = attn_dim,
                    causal = causal,
                    act = act,
                    act_in =  act_in,
                    act_out = act_out,
                    shift_size = shift_size,
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
