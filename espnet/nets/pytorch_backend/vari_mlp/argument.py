# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer common arguments."""


from distutils.util import strtobool


def add_arguments_mlp_common(group):
    """Add ComplexFNet common arguments."""
    group.add_argument(
        "--tiny_attn_dim",
        type=int,
        default=128,
        help="dim of tiny atttention",
    )
    group.add_argument(
        "--dim_fft",
        type=int,
        default=512,
        help="dim of FFT",
    )
    group.add_argument(
        "--causal",
        type=strtobool,
        default=False,
    )
    group.add_argument(
        "--act",
        choices=["", "gelu", "tanh", "sigmoid"],
        default="",
        help="type of activation layer",
    )
    group.add_argument(
        "--act_in",
        choices=["", "gelu", "tanh", "sigmoid"],
        default="",
        help="activation type of MLPBlock",
    )
    group.add_argument(
        "--act_out",
        choices=["", "gelu", "tanh", "sigmoid"],
        default="",
        help="activation type of MLPBlock",
    )
    group.add_argument(
        "--mlp_type",
        choices=["cmlp", "cmlp2", "tsmlp", "fmlp", "fmlp2", "fdmlp", "fdmlp2", "conv_cmlp", "conv_cmlp2", "conv_tsmlp"],
        default='cmlp',
        help="type of MLP based method",
    )
    group.add_argument(
        "--time_shift",
        default=1,
        type=int,
        help="time shift in tsmlp",
    )
    group.add_argument(
        "--mlp_module_kernel",
        default=15,
        type=int,
        help="kernel size of f-mlp/c-mlp",
    )
    group.add_argument(
        "--mlp_module_dropout_rate",
        default=0,
        type=float,
        help="dropout rate in gating unit",
    )

    """Add arguments for interCTC/self-cond CTC"""
    group.add_argument(
        "--multi-position",
        type=int,
        action="append",
        help="Positions of layers which take intermediate losses",
    )
    group.add_argument(
        "--toplayer-w",
        default=0.5,
        type=float,
        help="L := (1 − w) * L_toplayer + w * L_middlelayers",
    )

    """Add Transformer common arguments."""
    group.add_argument(
        "--transformer-init",
        type=str,
        default="pytorch",
        choices=[
            "pytorch",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        ],
        help="how to initialize transformer parameters",
    )
    group.add_argument(
        "--transformer-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "linear", "embed"],
        help="transformer input layer type",
    )
    group.add_argument(
        "--transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="dropout in transformer attention. use --dropout-rate if None is set",
    )
    group.add_argument(
        "--transformer-lr",
        default=10.0,
        type=float,
        help="Initial value of learning rate",
    )
    group.add_argument(
        "--transformer-warmup-steps",
        default=25000,
        type=int,
        help="optimizer warmup steps",
    )
    group.add_argument(
        "--transformer-length-normalized-loss",
        default=True,
        type=strtobool,
        help="normalize loss by length",
    )
    group.add_argument(
        "--transformer-encoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "rel_selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer encoder self-attention layer type",
    )
    group.add_argument(
        "--transformer-decoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer decoder self-attention layer type",
    )
    # Lightweight/Dynamic convolution related parameters.
    # See https://arxiv.org/abs/1912.11793v2
    # and https://arxiv.org/abs/1901.10430 for detail of the method.
    # Configurations used in the first paper are in
    # egs/{csj, librispeech}/asr1/conf/tuning/ld_conv/
    group.add_argument(
        "--wshare",
        default=4,
        type=int,
        help="Number of parameter shargin for lightweight convolution",
    )
    group.add_argument(
        "--ldconv-encoder-kernel-length",
        default="21_23_25_27_29_31_33_35_37_39_41_43",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Encoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--ldconv-decoder-kernel-length",
        default="11_13_15_17_19_21",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Decoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--ldconv-usebias",
        type=strtobool,
        default=False,
        help="use bias term in lightweight/dynamic convolution",
    )
    group.add_argument(
        "--dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for the encoder",
    )
    # Encoder
    group.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers (for shared recognition part "
        "in multi-speaker asr mode)",
    )
    group.add_argument(
        "--eunits",
        "-u",
        default=300,
        type=int,
        help="Number of encoder hidden units",
    )
    # Attention
    group.add_argument(
        "--adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    group.add_argument(
        "--aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    # Decoder
    group.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )
    return group
