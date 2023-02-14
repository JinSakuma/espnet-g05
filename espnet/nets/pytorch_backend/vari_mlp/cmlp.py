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


class ConvGatingUnit(nn.Module):
    def __init__(self, eunits, act = nn.Identity(), dropout=nn.Identity(), cnn_kernel=15, bias=True):
        super().__init__()
        dim_out = eunits // 2
        self.norm = nn.LayerNorm(dim_out)
        self.dropout = dropout
        self.depthwise_conv = nn.Conv1d(
                dim_out,
                dim_out,
                cnn_kernel,
                stride=1,
                padding=(cnn_kernel - 1) // 2,
                groups=dim_out,
                bias=bias,
                )

        self.act = act
        self.conv_act = nn.SiLU()

    def forward(self, x, tiny_attn = None):
        res, gate = x.chunk(2, dim = -1)
        
        gate = self.norm(gate)
        gate = gate.permute(0,2,1)
        gate = self.depthwise_conv(gate)
        gate = self.conv_act(gate)
        gate = self.dropout(gate)
        gate = gate.permute(0,2,1)

        if tiny_attn is not None:
            gate = gate + tiny_attn

        return self.act(gate) * res


class ConvGatingUnit2(nn.Module):
    def __init__(self, eunits, act=nn.Identity(), dropout=nn.Identity(), cnn_kernel=15, bias=True):
        super().__init__()
        dim_out = eunits // 2
        self.norm = nn.LayerNorm(dim_out)
        self.dropout = dropout
        self.depthwise_conv = nn.Conv1d(
                dim_out,
                dim_out,
                cnn_kernel,
                stride=1,
                padding=(cnn_kernel - 1) // 2,
                groups=dim_out,
                bias=bias,
                )

        self.pointwise_conv = nn.Conv1d(
                dim_out,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                )

        self.act = act
        self.conv_act = nn.SiLU()

    def forward(self, x, tiny_attn = None):
        res, gate = x.chunk(2, dim = -1)
        
        gate = self.norm(gate)
        gate = gate.permute(0,2,1)
        gate = self.depthwise_conv(gate)
        gate = self.conv_act(gate)
        gate = self.dropout(gate)
        gate = self.pointwise_conv(gate)
        gate = gate.permute(0,2,1)

        if tiny_attn is not None:
            gate = gate + tiny_attn

        return self.act(gate) * res


class CMLPBlock(nn.Module):
    def __init__(
        self,
        adim,
        eunits,
        act=nn.Identity(),
        act_in=nn.GELU(),
        act_out=nn.Identity(),
        attn_dim=0,
        causal=False,
        cmlp_type=1,
        kernel=15,
        dropout=0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(adim, eunits)
        self.proj_out = nn.Linear(eunits // 2, adim)
        self.act_in = act_in
        self.act_out = act_out
        self.pre_norm = nn.LayerNorm(adim)

        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()


        if cmlp_type == 1:
            self.cgu = ConvGatingUnit(eunits, act, self.dropout, cnn_kernel=kernel)
        elif cmlp_type == 2:
            self.cgu = ConvGatingUnit2(eunits, act, self.dropout, cnn_kernel=kernel)
        else:
            NotImplementedError

        self.attn = Attention(adim, eunits // 2, attn_dim, causal) if attn_dim>0 else None

    def forward(self, x, masks):
        residual = x
        
        x = self.pre_norm(x)

        tiny_attn = self.attn(x) if self.attn is not None else None
    
        x = self.proj_in(x)
        x = self.act_in(x)
        
        x = self.cgu(x, tiny_attn=tiny_attn)

        x = self.proj_out(x)
        x = self.act_out(x)

        return residual + x, masks


class  CMLPEncoder(nn.Module):
    def __init__(
        self,
        elayers,
        adim,
        eunits,
        act = nn.Identity(),
        act_in = nn.GELU(),
        act_out = nn.Identity(),
        attn_dim = 0,
        causal = False,
        cmlp_type = 1,
        kernel=15,
        dropout=0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                CMLPBlock(
                    adim = adim,
                    eunits = eunits,
                    attn_dim = attn_dim,
                    causal = causal,
                    act = act,
                    act_in = act_in,
                    act_out = act_out,
                    cmlp_type = cmlp_type,
                    kernel = kernel,
                    dropout = dropout,
                ) for i in range(elayers)
            ]
        )
        self.norm = nn.LayerNorm(adim)

    def forward(self, x):
        out = nn.Sequential(*self.layers)(x)
        out = self.norm(out)
        return out

