from typing import Union, Tuple, Type

from torch import nn

from .coord_conv import CoordConv


class ConvBNAct(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Activation layer.
        Default behaviour is Conv-BN-Act. To exclude Batchnorm module use
        `use_normalization=False`, to exclude activation use `activation_type=None`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.

    Configuration via advanced options:
    1) is_coord_conv: False by default, which means that only 2 channels will be added -
        a row index channel and a column index channel to the input view.
    2) 'is_coord_conv' = True, 'with_r' = True will add a 3rd distance-channel from the center to the input view.
    3) 'is_transpose_conv' = True (with additional parameter 'output_padding') means that instead of the usual convolution
        need to use the transposed one.

    The CoordConv is not supported with the DeepWise implementation (in most cases),
    so a restriction is introduced: one of the conditions 'is_coord_conv' = False or 'groups' = 1 must be met.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int]],
            activation_type: Type[nn.Module],
            stride: Union[int, Tuple[int, int]] = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            output_padding: Union[int, Tuple[int, int]] = 0,
            use_normalization: bool = True,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            activation_kwargs=None,
            is_coord_conv: bool = False,
            with_r: bool = False,
            is_transpose_conv: bool = False,
    ):
        if groups <= 0:
            raise ValueError('Groups must be a positive integer')
        if groups > 1 and is_coord_conv:
            raise ValueError('It is possible to choose only either CoordConv or DeepWise implementation.'
                             ' Set "is_coord_conv" = False or number of groups "groups" = 1')

        #ToDo - padding is str
        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {}

        self.seq = nn.Sequential()
        general_conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": stride,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype
        }

        if (not is_transpose_conv) and (not is_coord_conv):
            self.seq.add_module("conv", nn.Conv2d(**general_conv_kwargs))
        elif is_transpose_conv and (not is_coord_conv):
            self.seq.add_module("tpose_conv", nn.ConvTranspose2d(**general_conv_kwargs, output_padding=output_padding))
        else:  # is_coord_conv
            module_name = "coord_conv"
            if is_transpose_conv:
                module_name = "tpose_" + module_name
            self.seq.add_module(module_name, CoordConv(**general_conv_kwargs, output_padding=output_padding,
                                                       is_transpose_conv=is_transpose_conv, with_r=with_r))

        if use_normalization:
            self.seq.add_module(
                "bn",
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                               track_running_stats=track_running_stats, device=device, dtype=dtype),
            )
        if activation_type is not None:
            self.seq.add_module("act", activation_type(**activation_kwargs))

    def forward(self, x):
        return self.seq(x)
