import math

import numpy as np

import torch
from torch import nn

from torchsparse.sparse_tensor import *
from torchsparse.utils import make_list

from ..functional import *


__all__ = ['Conv3d', 'DeformConv3d', 'Conv4d', 'DepthwiseConv4d', 'DepthwiseConv3d', 'Conv3_5d']


class Conv3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 transpose=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = make_list(kernel_size)
        self.stride = make_list(stride)
        self.dilation = make_list(dilation)

        if np.prod(self.kernel_size) > 1:
            self.kernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, out_channels))
        else:
            assert not transpose
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
                      
        self.bias = None if not bias else nn.Parameter(torch.zeros(out_channels))

        self.t = transpose
        self.init_weight()
            

    def __repr__(self):
        if not self.t:
            return 'Conv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)
        else:
            return 'TransposedConv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)

    def init_weight(self):
        std = 1. / math.sqrt(self.out_channels if self.t else self.in_channels * np.prod(self.kernel_size))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return conv3d(inputs,
                      self.kernel,
                      self.bias,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)

'''
class XConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transpose: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels = in_channels
        self.out_channels = out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.kernel = nn.Parameter(torch.zeros(self.kernel_size ** 3, in_channels, out_channels)) if self.kernel_size > 1 else \
                      nn.Parameter(torch.zeros(in_channels, out_channels))
                      
        self.bias = None if not bias else nn.Parameter(torch.zeros(out_channels))
        self.t = transpose
        self.init_weight()

        if kernel_size == 1:
            assert not transpose

    def __repr__(self):
        if not self.t:
            return 'XConv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)
        else:
            return 'XConv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)

    def init_weight(self):
        std = 1. / math.sqrt(
            self.out_channels if self.t else self.in_channels *
            (self.kernel_size ** 3))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs, space):
        return xconv3d(inputs, space,
                      self.kernel,
                      self.bias,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)
'''

class DeformConv3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 transpose=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = make_list(kernel_size)
        self.stride = make_list(stride)
        self.dilation = make_list(dilation)

        assert(np.prod(kernel_size) > 1)
        assert(not transpose)

        self.kernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, out_channels))
        self.bias = None if not bias else nn.Parameter(torch.zeros(out_channels))

        self.pkernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, 3 * np.prod(self.kernel_size)))
        self.pkernel.register_hook(self._set_lr)

        self.init_weight()

    @staticmethod
    def _set_lr(grad):
        return grad * 0.1

    def __repr__(self):
        return 'DeformConv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.dilation)


    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels * np.prod(self.kernel_size))
        
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        self.pkernel.data.zero_()

    def forward(self, inputs):
        return deformconv3d(inputs,
                      self.kernel,
                      self.bias,
                      self.pkernel,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)


class Conv4d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 transpose=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = make_list(kernel_size, 4)
        self.stride = make_list(stride, 4)
        self.dilation = make_list(dilation, 4)

        if isinstance(self.kernel_size, str):
            if self.kernel_size == 'hypercross':
                self.kernel = nn.Parameter(torch.zeros(9, in_channels, out_channels))
            else:
                raise NotImplementedError
        elif np.prod(self.kernel_size) > 1:
            self.kernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, out_channels))
        else:
            assert not transpose
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
                      
        self.bias = None if not bias else nn.Parameter(torch.zeros(out_channels))
        self.t = transpose
        self.init_weight()

        if kernel_size == 1:
            assert not transpose

    def __repr__(self):
        if not self.t:
            return 'Conv4d(in_channels={}, out_channels={}, kernel_size={}, stride={}, dilation={})'.format(
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)
        else:
            return 'TransposedConv4d(in_channels={}, out_channels={}, kernel_size={}, stride={}, dilation={})'.format(
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)

    def init_weight(self):
        std = 1. / math.sqrt(self.out_channels if self.t else self.in_channels * self.kernel.shape[0])
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return conv4d(inputs,
                      self.kernel,
                      self.bias,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)


# channel-separated (depth-wise group conv)
class DepthwiseConv3d(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 transpose=False):

        super().__init__()
        self.in_channels = in_channels

        self.kernel_size = make_list(kernel_size)
        self.stride = make_list(stride)
        self.dilation = make_list(dilation)

        assert (np.prod(self.kernel_size) > 1)

        self.kernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, 1))              
        self.bias = None

        self.t = transpose

        self.init_weight()

    def __repr__(self):
        return 'DepthwiseConv3d(in_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
            self.in_channels, self.kernel_size,
            self.stride, self.dilation)
    

    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels * np.prod(self.kernel_size))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return dwconv3d(inputs,
                      self.kernel,
                      self.bias,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)

class DepthwiseConv4d(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 transpose=False):

        super().__init__()
        self.in_channels = in_channels

        self.kernel_size = make_list(kernel_size, 4)
        self.stride = make_list(stride, 4)
        self.dilation = make_list(dilation, 4)

        assert (np.prod(self.kernel_size) > 1)
        
        self.kernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, 1))              
        self.bias = None

        self.t = transpose

        self.init_weight()

    def __repr__(self):
        return 'DepthwiseConv4d(in_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
            self.in_channels, self.kernel_size,
            self.stride, self.dilation)
    

    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels * np.prod(self.kernel_size))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return dwconv4d(inputs,
                      self.kernel,
                      self.bias,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)


class Conv3_5d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 transpose=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        assert isinstance(dilation, int)

        self.kernel_size = np.array([kernel_size, kernel_size, kernel_size])
        self.stride = np.array([1, stride, stride, stride])
        self.dilation = np.array([1, dilation, dilation, dilation])

        if np.prod(self.kernel_size) > 1:
            self.kernel = nn.Parameter(torch.zeros(np.prod(self.kernel_size), in_channels, out_channels))
        else:
            assert not transpose
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
                      
        self.bias = None if not bias else nn.Parameter(torch.zeros(out_channels))

        self.t = transpose
        self.init_weight()
            

    def __repr__(self):
        if not self.t:
            return 'Conv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)
        else:
            return 'TransposedConv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)

    def init_weight(self):
        std = 1. / math.sqrt(self.out_channels if self.t else self.in_channels * np.prod(self.kernel_size))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return conv3_5d(inputs,
                      self.kernel,
                      self.bias,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)