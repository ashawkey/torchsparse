from torch.autograd import Function

import torchsparse_cuda

__all__ = ['sphash']


class HashGPU(Function):
    @staticmethod
    def forward(ctx, idx):
        if 'cuda' in str(idx.device):
            return torchsparse_cuda.hash_forward(idx.contiguous())
        elif 'cpu' in str(idx.device):
            return torchsparse_cuda.cpu_hash_forward(idx.int().contiguous())
        else:
            device = idx.device
            return torchsparse_cuda.cpu_hash_forward(idx.int().contiguous().cpu()).to(device)


class KernelHashGPU(Function):
    @staticmethod
    def forward(ctx, idx, koffset):
        if 'cuda' in str(idx.device):
            return torchsparse_cuda.kernel_hash_forward(idx.contiguous(), koffset.contiguous())
        elif 'cpu' in str(idx.device):
            return torchsparse_cuda.cpu_kernel_hash_forward(idx.int().contiguous(), koffset.int().contiguous())
        else:
            device = idx.device
            return torchsparse_cuda.cpu_kernel_hash_forward(idx.int().contiguous().cpu(), koffset.int().contiguous().cpu()).to(device)

class DeformKernelHashGPU(Function):
    @staticmethod
    def forward(ctx, idx, koffset):
        if 'cuda' in str(idx.device):
            return torchsparse_cuda.deform_kernel_hash_forward(idx.contiguous(), koffset.contiguous())
        elif 'cpu' in str(idx.device):
            raise NotImplementedError
        else:
            raise NotImplementedError
          
hash_gpu = HashGPU.apply
kernel_hash_gpu = KernelHashGPU.apply
deform_kernel_hash_gpu = DeformKernelHashGPU.apply


def sphash(idx, koffset=None):
    if koffset is None:
        return hash_gpu(idx)
    elif len(koffset.shape) == 2:
        return kernel_hash_gpu(idx, koffset)
    else:
        return deform_kernel_hash_gpu(idx, koffset)
