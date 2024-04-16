import torch
from torch import nn as nn
import torch.nn.functional as F
from utils_torch import *

class FlippedQuanv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(FlippedQuanv3x3, self).__init__()
        super(FlippedQuanv3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.pad_l, self.pad_r, self.pad_t, self.pad_b = padding if padding is not None else (0, 0, 0, 0)
        self.pad = (self.pad_l, self.pad_r, self.pad_t, self.pad_b)
        self.weight = torch.nn.Parameter(torch.randn((out_channels, in_channels, 4 ** 2 - 1)).type(COMPLEX_DTYPE))
        self.bias = torch.nn.Parameter(torch.randn((out_channels, 1)).type(COMPLEX_DTYPE))

    def forward(self, x):
        # x has shape (batchsize ,c_in, h, w)
        # weight has shape (c_out, c_in, 15)
        # bias has shape (c_out, 1)
        x = x.type(COMPLEX_DTYPE)
        c_in, h_in, w_in = x.shape[-3], x.shape[-2], x.shape[-1]
        patches = extract_patches(x, patch_size=3, stride=self.stride, padding=self.pad)
        h_out = (h_in - 3 + (self.pad_t + self.pad_b)) // self.stride + 1
        w_out = (w_in - 3 + (self.pad_l + self.pad_r)) // self.stride + 1

        out = vmap_vmap_single_kernel_op_through_extracted_patches(self.weight, patches)

        out = out + self.bias
        return out.reshape((-1, self.out_channels, h_out, w_out)).type(REAL_DTYPE)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}, padding={self.padding}'

