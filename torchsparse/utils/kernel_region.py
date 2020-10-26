import json
import os
import os.path
import random
import sys
import h5py
import numpy as np
import torch

from torchsparse.utils import make_list

__all__ = ['KernelRegion', 'KernelRegion4D']


class KernelRegion:
    def __init__(self,
                 kernel_size,
                 stride,
                 dilation):

        self.kernel_size = make_list(kernel_size, 3)
        self.stride = make_list(stride, 3)
        self.dilation = make_list(dilation, 3)

        x_offset = (np.arange(-self.kernel_size[0] // 2 + 1, self.kernel_size[0] // 2 + 1) * self.stride[0] * self.dilation[0]).tolist()
        y_offset = (np.arange(-self.kernel_size[1] // 2 + 1, self.kernel_size[1] // 2 + 1) * self.stride[1] * self.dilation[1]).tolist()
        z_offset = (np.arange(-self.kernel_size[2] // 2 + 1, self.kernel_size[2] // 2 + 1) * self.stride[2] * self.dilation[2]).tolist() 

        kernel_offset = np.array([[x, y, z] for z in z_offset for y in y_offset for x in x_offset])

        self.kernel_offset = torch.from_numpy(kernel_offset).int()

    def get_kernel_offset(self):
        return self.kernel_offset


class KernelRegion4D:
    def __init__(self,
                 kernel_size,
                 stride,
                 dilation,
                 ):

        if isinstance(kernel_size, str):
            if kernel_size == 'hypercross':
                self.stride = make_list(stride, 4)
                self.dilation = make_list(dilation, 4)
                kernel_offset = np.array([[0, 0, 0, 0], 
                                        [1, 0, 0, 0], [-1, 0, 0, 0], 
                                        [0, 1, 0, 0], [0, -1, 0, 0],
                                        [0, 0, 1, 0], [0, 0, -1, 0],
                                        [0, 0, 0, 1], [0, 0, 0, -1]])

                kernel_offset *= self.stride * self.dilation
            else:
                raise NotImplementedError
        
        else:
            self.kernel_size = make_list(kernel_size, 4)
            self.stride = make_list(stride, 4)
            self.dilation = make_list(dilation, 4)
            
            t_offset = (np.arange(-self.kernel_size[0] // 2 + 1, self.kernel_size[0] // 2 + 1) * self.stride[0] * self.dilation[0]).tolist() 
            x_offset = (np.arange(-self.kernel_size[1] // 2 + 1, self.kernel_size[1] // 2 + 1) * self.stride[1] * self.dilation[1]).tolist()
            y_offset = (np.arange(-self.kernel_size[2] // 2 + 1, self.kernel_size[2] // 2 + 1) * self.stride[2] * self.dilation[2]).tolist()
            z_offset = (np.arange(-self.kernel_size[3] // 2 + 1, self.kernel_size[3] // 2 + 1) * self.stride[3] * self.dilation[3]).tolist() 

            kernel_offset = np.array([[t, x, y, z] for t in t_offset for z in z_offset for y in y_offset for x in x_offset])

        self.kernel_offset = torch.from_numpy(kernel_offset).int()

    def get_kernel_offset(self):
        return self.kernel_offset