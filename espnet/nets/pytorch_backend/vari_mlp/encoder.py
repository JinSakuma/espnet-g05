# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch

from espnet.nets.pytorch_backend.vari_mlp.cmlp import CMLPBlock
from espnet.nets.pytorch_backend.vari_mlp.conv_cmlp import ConvCMLPBlock
from espnet.nets.pytorch_backend.vari_mlp.tsmlp import TSMLPBlock
from espnet.nets.pytorch_backend.vari_mlp.conv_tsmlp import ConvTSMLPBlock
from espnet.nets.pytorch_backend.vari_mlp.fmlp import FMLPBlock
from espnet.nets.pytorch_backend.vari_mlp.fmlp_gmlp_type import FMLPBlock2
from espnet.nets.pytorch_backend.vari_mlp.fdmlp import FDMLPBlock

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.repeat import repeat


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.
    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in self_attention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        selfattention_layer_type (str): Encoder attention layer type.
        padding_idx (int): Padding idx for input_layer=embed.
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length="11",
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        selfattention_layer_type="selfattn",
        padding_idx=-1,
        dim_fft=512,
        mlp_type="cmlp",
        attn_dim=0,
        causal=False,
        act="",
        act_in="",
        act_out="",
        shift_size=1,
        mlp_module_kernel=15,
        mlp_module_dropout_rate=0,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 8
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
       
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        if mlp_type == "cmlp" or mlp_type == "cmlp2":
            if mlp_type == "cmlp":
                cmlp_type=1
            else:
                cmlp_type=2

            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: CMLPBlock(
                        adim=attention_dim,
                        eunits=linear_units,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        cmlp_type=cmlp_type,
                        kernel=mlp_module_kernel,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        if mlp_type == "conv_cmlp" or mlp_type == "conv_cmlp2":
            if mlp_type == "conv_cmlp":
                cmlp_type=1
            else:
                cmlp_type=2

            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: ConvCMLPBlock(
                        adim=attention_dim,
                        eunits=linear_units,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        cmlp_type=cmlp_type,
                        kernel=mlp_module_kernel,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        elif mlp_type == "tsmlp":
            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: TSMLPBlock(
                        adim=attention_dim,
                        eunits=linear_units,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        shift_size=shift_size,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        elif mlp_type == "conv_tsmlp":
            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: ConvTSMLPBlock(
                        adim=attention_dim,
                        eunits=linear_units,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        shift_size=shift_size,
                        kernel=mlp_module_kernel,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        elif mlp_type == "fmlp":
            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: FMLPBlock(
                        adim=attention_dim,
                        eunits=linear_units,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        kernel=mlp_module_kernel,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        elif mlp_type == "fmlp2":
            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: FMLPBlock2(
                        adim=attention_dim,
                        eunits=linear_units,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        kernel=mlp_module_kernel,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        elif mlp_type == "fdmlp" or mlp_type == "fdmlp2":
            if mlp_type == "fdmlp":
                fdmlp_type=1
            else:
                fdmlp_type=2

            self.encoders = repeat(
                    num_blocks,
                    lambda lnum: FDMLPBlock(
                        adim=attention_dim,
                        eunits=linear_units,
                        dim_fft=dim_fft,
                        act=self.str2act(act),
                        act_in=self.str2act(act_in),
                        act_out=self.str2act(act_out),
                        attn_dim=attn_dim,
                        causal=causal,
                        fdmlp_type=fdmlp_type,
                        dropout=mlp_module_dropout_rate,
                    ),
            )
        else:
            raise NotImplementedError

    def str2act(self, act):
        if act == "":
            activation = torch.nn.Identity()
        elif act == "tanh":
            activation = torch.nn.Tanh()
        elif act == "gelu":
            activation = torch.nn.GELU()
        elif act == "sigmoid":
            activation = torch.nn.Sigmoid() 
        else:
            NotImplementedError

        return activation
    
    def forward(self, xs, masks):
        """Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        xs = self.encoders(xs)
        # xs, masks = self.encoders(xs, masks)
        
        if self.normalize_before:
            xs = self.after_norm(xs)
        
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.
        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.
        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
