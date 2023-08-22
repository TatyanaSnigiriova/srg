import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from typing import Union, Tuple

"""
Paper:
    https://arxiv.org/pdf/1807.03247.pdf
    
This implementation is taken from official git repo:
https://github.com/walsvid/CoordConv/blob/master/coordconv.py

MIT License

Copyright (c) 2018 Walsvid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class AddCoords(nn.Module):
    def __init__(self, rank: int, with_r: bool = False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """

        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape

            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=input_tensor.dtype, device=input_tensor.get_device())
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=input_tensor.dtype, device=input_tensor.get_device())

            xx_range = torch.arange(dim_y, dtype=input_tensor.dtype, device=input_tensor.get_device())
            yy_range = torch.arange(dim_x, dtype=input_tensor.dtype, device=input_tensor.get_device())
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel / (dim_y - 1)
            yy_channel = yy_channel / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        else:
            raise NotImplementedError

        return out


class CoordConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            output_padding: Union[int, Tuple[int, int]] = 0,
            device=None,
            dtype=None,
            with_r: bool = False,
            is_transpose_conv: bool = False,
    ):
        super().__init__()

        self.rank = 2
        self.kwards = {
            "in_channels": in_channels + self.rank + int(with_r),
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype
        }
        self.addcoords = AddCoords(self.rank, with_r)
        if not is_transpose_conv:
            self.conv = nn.Conv2d(**self.kwards)
        else:
            self.conv = nn.ConvTranspose2d(**self.kwards, output_padding=output_padding)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
