import torch
import torch.fft
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


class FourierFilterUnit(nn.Module):
    def __init__(self, adim, act = nn.Identity(), kernel=15, attn=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(adim, kernel, dtype=torch.float32)*0.02)
        self.norm = nn.LayerNorm(adim)
        self.act = act
        self.attn = attn

    def forward(self, x):

        x = self.norm(x)
        n = x.shape[1]

        if self.attn is not None:
            tiny_attn = self.attn(x)

        x = x.permute(0,2,1)
        x = torch.fft.rfft(x, dim=-1, n=n)
        weight = torch.fft.rfft(self.weight, dim=-1, n=n)

        real = x.real*weight.real + x.imag*weight.imag
        imag = x.imag*weight.real - x.real*weight.imag
        x = torch.view_as_complex(torch.cat([real[..., None], imag[..., None]], dim=-1))

        x = torch.fft.irfft(x, n=n, dim=-1).permute(0,2,1)
        x = x[:, :n, :]

        if self.attn is not None:
            x = x + tiny_attn

        return self.act(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., act_in=nn.GELU(), act_out=nn.Identity()):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_in,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            act_out,
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class FMLPBlock(nn.Module):
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

        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        attn = Attention(adim, adim, attn_dim, causal) if attn_dim>0 else None 
        self.ffu = FourierFilterUnit(adim, act, kernel=15, attn=attn)
        self.ffn = FeedForward(adim, eunits, dropout=dropout, act_in=act_in, act_out=act_out)

    def forward(self, x):

        x = self.ffu(x) + x
        x = self.ffn(x) + x

        return x


class  FMLPEncoder(nn.Module):
    def __init__(
        self,
        elayers,
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
        
        self.layers = nn.ModuleList(
            [
                FMLPBlock(
                    adim = adim,
                    eunits = eunits,
                    act = act,
                    act_in = act_in,
                    act_out = act_out,
                    attn_dim = attn_dim,
                    causal=causal,
                    kernel=kernel,
                    dropout=dropout,
                ) for i in range(elayers)
            ]
        )
        self.norm = nn.LayerNorm(adim)

    def forward(self, x):
        out = nn.Sequential(*self.layers)(x)
        out = self.norm(out)
        return out
