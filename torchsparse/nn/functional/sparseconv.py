import copy

import numpy as np
import torch
from torch.autograd import Function

import torchsparse_cuda
from torchsparse import *
from torchsparse.nn.functional.convert_neighbor_map import *
from torchsparse.nn.functional.downsample import *
from torchsparse.nn.functional.hash import *
from torchsparse.nn.functional.query import *
from torchsparse.utils.kernel_region import *

__all__ = ['conv3d', 'deformconv3d', 'conv4d', 'dwconv4d', 'dwconv3d']


class SpConvolution(Function):
    @staticmethod
    def forward(ctx,
                features,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose=False):
        r"""
        features : torch.FloatTensor (N, c_in) Features of the input point cloud.
        kernel : torch.FloatTensor (K, c_in, c_out) Kernel. with K to be kernel volume.
        neighbor_map: torch.IntTensor (N_nonzero, 2) from -> to 
        return: (N', c_out).
        """
        features = features.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros(sizes[1], kernel.size(-1), device=features.device)
        else:
            out = torch.zeros(sizes[0], kernel.size(-1), device=features.device)

        if 'cuda' in str(features.device):
            torchsparse_cuda.sparseconv_forward(features, out, kernel, neighbor_map, neighbor_offset, transpose)
        else:
            raise NotImplementedError

        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset, transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):

        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        N_out = grad_out.size(0)
        grad_features = torch.zeros(N_in, c_in, device=features.device)
        grad_kernel = torch.zeros(K, c_in, c_out, device=kernel.device)

        if 'cuda' in str(features.device):
            torchsparse_cuda.sparseconv_backward(features, grad_features,
                                                 grad_out.contiguous(), kernel,
                                                 grad_kernel, neighbor_map,
                                                 neighbor_offset, transpose)
        else:
            raise NotImplementedError
        return grad_features, grad_kernel, None, None, None, None


sparseconv_op = SpConvolution.apply

def conv3d(inputs, kernel, bias, kernel_size, stride, dilation, transpose):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    # FC layer
    if len(kernel.shape) == 2:
        output_features = features.matmul(kernel)
        if bias is not None:
            output_features += bias

        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        output_tensor.check()

    # Conv layer
    elif not transpose:
        kernel_map = inputs.kernel_maps.get(f'{kernel_size}_{cur_stride}_{stride}_{dilation}', None)

        if kernel_map is None:
            kRegion = KernelRegion(kernel_size=kernel_size, stride=cur_stride, dilation=dilation)
            kOffset = kRegion.get_kernel_offset().to(features.device)

            new_coords = coords if np.prod(stride) == 1 else spdownsample(coords, stride * cur_stride) # FIXME
            hash_query = sphash(new_coords, kOffset)
            hash_target = sphash(coords)
            idx_query = sphashquery(hash_query, hash_target)

            idx_query = list(convert_neighbor_map_gpu(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = ((idx_query[0][:, 0].max() + 1).item(), (idx_query[0][:, 1].max() + 1).item())

            output_features = sparseconv_op(features, kernel, idx_query[0], idx_query[1], sizes, transpose)

            if bias is not None:
                output_features += bias

            output_tensor = SparseTensor(output_features, new_coords, cur_stride * stride)
            output_tensor.coord_maps = copy.deepcopy(inputs.coord_maps)
            output_tensor.check()
            output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
            output_tensor.kernel_maps[f'{kernel_size}_{cur_stride}_{stride}_{dilation}'] = idx_query + [sizes]

        else:
            output_features = sparseconv_op(features, kernel, kernel_map[0], kernel_map[1], kernel_map[2], transpose)

            if bias is not None:
                output_features += bias

            output_tensor = SparseTensor(output_features, coords, cur_stride)
            output_tensor.coord_maps = inputs.coord_maps
            output_tensor.check()
            output_tensor.kernel_maps = inputs.kernel_maps

    # TransposedConv layer
    else:
        original_stride = (cur_stride / stride).astype(int)
        kernel_map = inputs.kernel_maps.get(f'{kernel_size}_{original_stride}_{stride}_{dilation}', None) # assert not None ?

        output_features = sparseconv_op(features, kernel, kernel_map[0], kernel_map[1], kernel_map[2], transpose)
        if bias is not None:
            output_features += bias

        output_tensor = SparseTensor(output_features, inputs.coord_maps[str(original_stride)], original_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps

    return output_tensor

def deformconv3d(inputs, kernel, bias, pkernel, kernel_size, stride, dilation, transpose):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    # pconv
    kRegion = KernelRegion(kernel_size=kernel_size, stride=cur_stride, dilation=dilation)
    kOffset = kRegion.get_kernel_offset().to(features.device)

    new_coords = coords if np.prod(stride) == 1 else spdownsample(coords, stride * cur_stride)

    hash_query = sphash(new_coords, kOffset)
    hash_target = sphash(coords)
    idx_query = sphashquery(hash_query, hash_target) # [len(hash_query),]
    idx_query = list(convert_neighbor_map_gpu(idx_query)) # (neighbor map, neighbor offset)
    idx_query[1] = idx_query[1].to('cpu')
    sizes = ((idx_query[0][:, 0].max() + 1).item(), inputs.C.shape[0])
    
    kOffset2 = sparseconv_op(features, pkernel, idx_query[0], idx_query[1], sizes, transpose) # [N, 3*K]
    kOffset2 = torch.floor(kOffset2).int().view(-1, kernel.shape[0], 3) # [N, K, 3]
    kOffset2 = kOffset2 + kOffset.unsqueeze(0)

    # conv
    hash_query = sphash(new_coords, kOffset2)
    hash_target = sphash(coords)
    idx_query = sphashquery(hash_query, hash_target)
    idx_query = list(convert_neighbor_map_gpu(idx_query)) # [neighbor map, neighbor offset]
    idx_query[1] = idx_query[1].to('cpu')
    sizes = ((idx_query[0][:, 0].max() + 1).item(), (idx_query[0][:, 1].max() + 1).item())
    
    output_features = sparseconv_op(features, kernel, idx_query[0], idx_query[1], sizes, transpose) # [N, cout]

    if bias is not None:
        output_features += bias

    output_tensor = SparseTensor(output_features, new_coords, cur_stride * stride)
    output_tensor.coord_maps = inputs.coord_maps
    output_tensor.check()
    output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
    output_tensor.kernel_maps[f'{kernel_size}_{cur_stride}_{stride}_{dilation}'] = idx_query + [sizes]

    return output_tensor


def conv4d(inputs, kernel, bias, kernel_size, stride, dilation, transpose):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    # FC layer
    if len(kernel.shape) == 2:
        output_features = features.matmul(kernel)
        if bias is not None:
            output_features += bias

        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        output_tensor.check()

    # Conv layer
    elif not transpose:
        kernel_map = inputs.kernel_maps.get(f'{kernel_size}_{cur_stride}_{stride}_{dilation}', None)

        if kernel_map is None:
            kRegion = KernelRegion4D(kernel_size=kernel_size, stride=cur_stride, dilation=dilation)
            kOffset = kRegion.get_kernel_offset().to(features.device)

            new_coords = coords if np.prod(stride) == 1 else spdownsample(coords, stride * cur_stride)
            hash_query = sphash(new_coords, kOffset)
            hash_target = sphash(coords)
            idx_query = sphashquery(hash_query, hash_target)

            idx_query = list(convert_neighbor_map_gpu(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = ((idx_query[0][:, 0].max() + 1).item(), (idx_query[0][:, 1].max() + 1).item())

            output_features = sparseconv_op(features, kernel, idx_query[0], idx_query[1], sizes, transpose)

            if bias is not None:
                output_features += bias

            output_tensor = SparseTensor(output_features, new_coords, cur_stride * stride)
            output_tensor.coord_maps = copy.deepcopy(inputs.coord_maps)
            output_tensor.check()
            output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
            output_tensor.kernel_maps[f'{kernel_size}_{cur_stride}_{stride}_{dilation}'] = idx_query + [sizes]

        else:
            output_features = sparseconv_op(features, kernel, kernel_map[0], kernel_map[1], kernel_map[2], transpose)

            if bias is not None:
                output_features += bias

            output_tensor = SparseTensor(output_features, coords, cur_stride)
            output_tensor.coord_maps = inputs.coord_maps
            output_tensor.check()
            output_tensor.kernel_maps = inputs.kernel_maps

    # TransposedConv layer
    else:
        original_stride = (cur_stride / stride).astype(int)
        kernel_map = inputs.kernel_maps.get(f'{kernel_size}_{original_stride}_{stride}_{dilation}', None) # assert not None ?

        output_features = sparseconv_op(features, kernel, kernel_map[0], kernel_map[1], kernel_map[2], transpose)
        if bias is not None:
            output_features += bias

        output_tensor = SparseTensor(output_features, inputs.coord_maps[str(original_stride)], original_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps

    return output_tensor

################
'''
class SpXConvolution(Function):
    @staticmethod
    def forward(ctx,
                features,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose=False):

        # assert transpose == False

        features = features.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros(sizes[1], kernel.size(-1), device=features.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(sizes[0], kernel.size(-1), device=features.device)

        if 'cuda' in str(features.device):
            torchsparse_cuda.sparseconv_forward(features, out, kernel, neighbor_map, neighbor_offset, transpose)
        #elif 'cpu' in str(features.device):
        #    torchsparse_cuda.sparseconv_cpu_forward(features, out, kernel, neighbor_map, neighbor_offset.cpu(), transpose)
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(kernel.shape[0]):
                cur_ed = cur_st + neighbor_offset[kernel_idx]
                in_map = neighbor_map[cur_st:cur_ed, 0].long()
                out_map = neighbor_map[cur_st:cur_ed, 1].long()
                cur_st += neighbor_offset[kernel_idx]

                if transpose:
                    in_map, out_map = out_map, in_map
                # gather
                cur_feat = features[in_map]
                # gemm
                cur_feat = torch.mm(cur_feat, kernel[kernel_idx])
                # scatter
                out[out_map] += cur_feat

        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset, transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):

        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        N_out = grad_out.size(0)
        grad_features = torch.zeros(N_in, c_in, device=features.device)
        grad_kernel = torch.zeros(K, c_in, c_out, device=kernel.device)

        if 'cuda' in str(features.device):
            torchsparse_cuda.sparseconv_backward(features, grad_features,
                                                 grad_out.contiguous(), kernel,
                                                 grad_kernel, neighbor_map,
                                                 neighbor_offset, transpose)
        else:
            raise NotImplementedError
        return grad_features, grad_kernel, None, None, None, None


sparsexconv_op = SpXConvolution.apply

def xconv3d(inputs, space, kernel, bias=None, stride=1, dilation=1, transpose=False):
    # assert stride == 1
    # assert ks > 1
    # assert transpose == False

    ks = 1 if len(kernel.shape) == 2 else int(round(kernel.shape[0] ** (1 / 3.)))

    kRegion = KernelRegion(kernel_size=ks, tensor_stride=inputs.s)
    kOffset = kRegion.get_kernel_offset().to(inputs.F.device)

    hash_query = sphash(inputs.C, kOffset) # [ks**3, inputs.C.shape[0]]
    hash_target = sphash(space.C) # [space.C.shape[0]] 
    idx_query = sphashquery(hash_query, hash_target) # [ks**3, inputs.C.shape[0]] idx of query point in target points.
    
    idx_query = list(convert_neighbor_map_gpu(idx_query)) # [neighbor map (N_nonzero, 2), neighbor offset]
    idx_query[1] = idx_query[1].to('cpu')
    # do not shrink on nonzero points.
    sizes = ((idx_query[0][:, 0].max() + 1).item(), inputs.C.shape[0])
    
    output_features = sparsexconv_op(space.F, kernel, idx_query[0], idx_query[1], sizes, transpose)

    if bias is not None:
        output_features += bias

    output_tensor = SparseTensor(output_features, inputs.C, inputs.s)
    output_tensor.coord_maps = inputs.coord_maps
    output_tensor.check()
    output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
    output_tensor.kernel_maps['k%d_os%d_s%d_d%d' % (ks, inputs.s, stride, dilation)] = idx_query + [sizes]
    
    return output_tensor
'''
################

class DwConvolution(Function):
    @staticmethod
    def forward(ctx,
                features,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose=False):
        features = features.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros((sizes[1], kernel.size(1)), device=features.device)
        else:
            out = torch.zeros((sizes[0], kernel.size(1)), device=features.device)

        if 'cuda' in str(features.device):
            torchsparse_cuda.sparsedwconv_forward(features, out, kernel, neighbor_map, neighbor_offset, transpose)
        else:
            raise NotImplementedError

        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset, transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size() # c_out == 1
        N_in = features.size(0)
        N_out = grad_out.size(0)
        grad_features = torch.zeros((N_in, c_in), device=features.device)
        grad_kernel = torch.zeros((K, c_in, c_out), device=kernel.device)

        if 'cuda' in str(features.device):
            torchsparse_cuda.sparsedwconv_backward(features, grad_features, grad_out.contiguous(), kernel, grad_kernel, neighbor_map, neighbor_offset, transpose)
        else:
            raise NotImplementedError

        return grad_features, grad_kernel, None, None, None, None


dwconv_op = DwConvolution.apply


def dwconv3d(inputs, kernel, bias, kernel_size, stride, dilation, transpose):
    features = inputs.F # [N, cin]
    coords = inputs.C
    cur_stride = inputs.s

    # conv
    if not transpose:
        kRegion = KernelRegion(kernel_size=kernel_size, stride=cur_stride, dilation=dilation)
        kOffset = kRegion.get_kernel_offset().to(features.device)

        new_coords = coords if np.prod(stride) == 1 else spdownsample(coords, stride * cur_stride)
        hash_query = sphash(new_coords, kOffset)
        hash_target = sphash(coords)
        idx_query = sphashquery(hash_query, hash_target)
        idx_query = list(convert_neighbor_map_gpu(idx_query)) # [neighbor map, neighbor offset]
        idx_query[1] = idx_query[1].to('cpu')
        sizes = ((idx_query[0][:, 0].max() + 1).item(), (idx_query[0][:, 1].max() + 1).item())

        output_features = dwconv_op(features, kernel, idx_query[0], idx_query[1], sizes, transpose)

        output_tensor = SparseTensor(output_features, new_coords, cur_stride * stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.check()
        output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
        output_tensor.kernel_maps[f'{kernel_size}_{cur_stride}_{stride}_{dilation}'] = idx_query + [sizes]
    # TransposedConv layer
    else:
        original_stride = (cur_stride / stride).astype(int)
        kernel_map = inputs.kernel_maps.get(f'{kernel_size}_{original_stride}_{stride}_{dilation}', None) # assert not None ?

        output_features = dwconv_op(features, kernel, kernel_map[0], kernel_map[1], kernel_map[2], transpose)

        output_tensor = SparseTensor(output_features, inputs.coord_maps[str(original_stride)], original_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps

    return output_tensor

def dwconv4d(inputs, kernel, bias, kernel_size, stride, dilation, transpose):
    features = inputs.F # [N, cin]
    coords = inputs.C
    cur_stride = inputs.s

    # conv
    if not transpose:
        kRegion = KernelRegion4D(kernel_size=kernel_size, stride=cur_stride, dilation=dilation)
        kOffset = kRegion.get_kernel_offset().to(features.device)

        new_coords = coords if np.prod(stride) == 1 else spdownsample(coords, stride * cur_stride)
        hash_query = sphash(new_coords, kOffset)
        hash_target = sphash(coords)
        idx_query = sphashquery(hash_query, hash_target)
        idx_query = list(convert_neighbor_map_gpu(idx_query)) # [neighbor map, neighbor offset]
        idx_query[1] = idx_query[1].to('cpu')
        sizes = ((idx_query[0][:, 0].max() + 1).item(), (idx_query[0][:, 1].max() + 1).item())

        output_features = dwconv_op(features, kernel, idx_query[0], idx_query[1], sizes, transpose)

        output_tensor = SparseTensor(output_features, new_coords, cur_stride * stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.check()
        output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
        output_tensor.kernel_maps[f'{kernel_size}_{cur_stride}_{stride}_{dilation}'] = idx_query + [sizes]
    # TransposedConv layer
    else:
        original_stride = (cur_stride / stride).astype(int)
        kernel_map = inputs.kernel_maps.get(f'{kernel_size}_{original_stride}_{stride}_{dilation}', None) # assert not None ?

        output_features = dwconv_op(features, kernel, kernel_map[0], kernel_map[1], kernel_map[2], transpose)

        output_tensor = SparseTensor(output_features, inputs.coord_maps[str(original_stride)], original_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps

    return output_tensor