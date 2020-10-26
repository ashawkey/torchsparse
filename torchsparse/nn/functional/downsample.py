import torch
from torch.autograd import Function

import torchsparse_cuda
from torchsparse.nn.functional.hash import *

__all__ = ['spdownsample', 'spdownsample3_5d']


class DownsampleGPU(Function):
    @staticmethod
    def forward(ctx, coords, ratio):
        
        coords_float = coords[:, :-1].float()
        ratio = torch.from_numpy(ratio).float().to(coords.device)
        coords_new = torch.floor(torch.floor(coords_float / ratio) * ratio).int() 
        coords_new = torch.cat([coords_new, coords[:, -1].view(-1, 1)], 1)
        coords_new_hash = sphash(coords_new)
        
        uq, inv, cnt = torch.unique(coords_new_hash, return_inverse=True, return_counts=True)
        inv = inv.int()
        cnt = cnt.int()

        if 'cuda' in str(coords.device):
            uq_coords = torch.round(torchsparse_cuda.insertion_forward(coords_new.float(), inv, cnt))
        elif 'cpu' in str(coords.device):
            uq_coords = torch.round(torchsparse_cuda.cpu_insertion_forward(coords_new.float(), inv, cnt))
        else:
            device = coords.device
            uq_coords = torch.round(torchsparse_cuda.cpu_insertion_forward(coords_new.float().cpu(), inv.cpu(), cnt.cpu()))
            uq_coords = uq_coords.to(device)

        uq_coords = uq_coords.int()

        return uq_coords

downsample_gpu = DownsampleGPU.apply

class DownsampleGPU3_5d(Function):
    @staticmethod
    def forward(ctx, coords, ratio):
        
        coords_float = coords[:, :-1].float()
        ratio = torch.from_numpy(ratio).float().to(coords.device)
        coords_new = torch.floor(torch.floor(coords_float / ratio) * ratio).int() 
        coords_new = torch.cat([coords_new, coords[:, -1].view(-1, 1)], 1)
        coords_new_hash = sphash(coords_new[:, 1:])
        
        uq, inv, cnt = torch.unique(coords_new_hash, return_inverse=True, return_counts=True)
        inv = inv.int()
        cnt = cnt.int()

        if 'cuda' in str(coords.device):
            uq_coords = torch.round(torchsparse_cuda.insertion_forward(coords_new.float(), inv, cnt))
        elif 'cpu' in str(coords.device):
            uq_coords = torch.round(torchsparse_cuda.cpu_insertion_forward(coords_new.float(), inv, cnt))
        else:
            device = coords.device
            uq_coords = torch.round(torchsparse_cuda.cpu_insertion_forward(coords_new.float().cpu(), inv.cpu(), cnt.cpu()))
            uq_coords = uq_coords.to(device)
            
        uq_coords = uq_coords.int()

        return uq_coords

downsample_gpu3_5d = DownsampleGPU3_5d.apply

def spdownsample(coords, ratio):
    return downsample_gpu(coords, ratio)

def spdownsample3_5d(coords, ratio):
    return downsample_gpu3_5d(coords, ratio)

