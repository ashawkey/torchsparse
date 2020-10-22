import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

import torchsparse_cuda

__all__ = ['spdevoxelize', 'calc_ti_weights', 'calc_quad_weights']


def calc_ti_weights(pc, idx_query, scale):
    # TBD: normalize the weights to a probability distribution. Note that some indices are "-1".
    with torch.no_grad():
        # don't want points to lie exactly on grid
        pc_grid = pc
        # don't use np.floor then convert to torch. numerical errors.
        scale = torch.from_numpy(scale).int().to(pc.device)
        pc_floor = torch.floor(pc / scale) * scale
        pc_ceil = pc_floor + scale

        pc_gridx = pc_grid[:, 0].view(-1, 1)
        pc_gridy = pc_grid[:, 1].view(-1, 1)
        pc_gridz = pc_grid[:, 2].view(-1, 1)
        pc_floorx = pc_floor[:, 0].view(-1, 1)
        pc_floory = pc_floor[:, 1].view(-1, 1)
        pc_floorz = pc_floor[:, 2].view(-1, 1)
        pc_ceilx = pc_ceil[:, 0].view(-1, 1)
        pc_ceily = pc_ceil[:, 1].view(-1, 1)
        pc_ceilz = pc_ceil[:, 2].view(-1, 1)
        pc_floorx = pc_floorx.float()
        pc_floory = pc_floory.float()
        pc_floorz = pc_floorz.float()
        pc_ceilx = pc_ceilx.float()
        pc_ceily = pc_ceily.float()
        pc_ceilz = pc_ceilz.float()
        weight000 = (pc_ceilx - pc_gridx) * (pc_ceily - pc_gridy) * (pc_ceilz - pc_gridz)
        weight001 = (pc_ceilx - pc_gridx) * (pc_ceily - pc_gridy) * (pc_gridz - pc_floorz)
        weight010 = (pc_ceilx - pc_gridx) * (pc_gridy - pc_floory) * (pc_ceilz - pc_gridz)
        weight011 = (pc_ceilx - pc_gridx) * (pc_gridy - pc_floory) * (pc_gridz - pc_floorz)
        weight100 = (pc_gridx - pc_floorx) * (pc_ceily - pc_gridy) * (pc_ceilz - pc_gridz)
        weight101 = (pc_gridx - pc_floorx) * (pc_ceily - pc_gridy) * (pc_gridz - pc_floorz)
        weight110 = (pc_gridx - pc_floorx) * (pc_gridy - pc_floory) * (pc_ceilz - pc_gridz)
        weight111 = (pc_gridx - pc_floorx) * (pc_gridy - pc_floory) * (pc_gridz - pc_floorz)

        all_weights = torch.cat([
            weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111
        ], 1).transpose(1, 0).contiguous() # [8, N]


        all_weights /= torch.prod(scale)

        all_weights[idx_query == -1] = 0
        all_weights /= all_weights.sum(0) + 1e-8

    return all_weights


def calc_quad_weights(pc, idx_query, scale):
    # TBD: normalize the weights to a probability distribution. Note that some indices are "-1".
    with torch.no_grad():
        # don't want points to lie exactly on grid
        pc_grid = pc
        # don't use np.floor then convert to torch. numerical errors.
        scale = torch.from_numpy(scale).int().to(pc.device)
        pc_floor = torch.floor(pc / scale) * scale
        pc_ceil = pc_floor + scale

        pc_gridt = pc_grid[:, 0].view(-1, 1)
        pc_gridx = pc_grid[:, 1].view(-1, 1)
        pc_gridy = pc_grid[:, 2].view(-1, 1)
        pc_gridz = pc_grid[:, 3].view(-1, 1)

        pc_floort = pc_floor[:, 0].view(-1, 1).float()
        pc_floorx = pc_floor[:, 1].view(-1, 1).float()
        pc_floory = pc_floor[:, 2].view(-1, 1).float()
        pc_floorz = pc_floor[:, 3].view(-1, 1).float()

        pc_ceilt = pc_ceil[:, 0].view(-1, 1).float()
        pc_ceilx = pc_ceil[:, 1].view(-1, 1).float()
        pc_ceily = pc_ceil[:, 2].view(-1, 1).float()
        pc_ceilz = pc_ceil[:, 3].view(-1, 1).float()

        weight0000 = (pc_ceilt - pc_gridt) * (pc_ceilx - pc_gridx) * (pc_ceily - pc_gridy) * (pc_ceilz - pc_gridz)
        weight0001 = (pc_ceilt - pc_gridt) * (pc_ceilx - pc_gridx) * (pc_ceily - pc_gridy) * (pc_gridz - pc_floorz)
        weight0010 = (pc_ceilt - pc_gridt) * (pc_ceilx - pc_gridx) * (pc_gridy - pc_floory) * (pc_ceilz - pc_gridz)
        weight0011 = (pc_ceilt - pc_gridt) * (pc_ceilx - pc_gridx) * (pc_gridy - pc_floory) * (pc_gridz - pc_floorz)
        weight0100 = (pc_ceilt - pc_gridt) * (pc_gridx - pc_floorx) * (pc_ceily - pc_gridy) * (pc_ceilz - pc_gridz)
        weight0101 = (pc_ceilt - pc_gridt) * (pc_gridx - pc_floorx) * (pc_ceily - pc_gridy) * (pc_gridz - pc_floorz)
        weight0110 = (pc_ceilt - pc_gridt) * (pc_gridx - pc_floorx) * (pc_gridy - pc_floory) * (pc_ceilz - pc_gridz)
        weight0111 = (pc_ceilt - pc_gridt) * (pc_gridx - pc_floorx) * (pc_gridy - pc_floory) * (pc_gridz - pc_floorz)

        weight1000 = (pc_gridt - pc_floort) * (pc_ceilx - pc_gridx) * (pc_ceily - pc_gridy) * (pc_ceilz - pc_gridz)
        weight1001 = (pc_gridt - pc_floort) * (pc_ceilx - pc_gridx) * (pc_ceily - pc_gridy) * (pc_gridz - pc_floorz)
        weight1010 = (pc_gridt - pc_floort) * (pc_ceilx - pc_gridx) * (pc_gridy - pc_floory) * (pc_ceilz - pc_gridz)
        weight1011 = (pc_gridt - pc_floort) * (pc_ceilx - pc_gridx) * (pc_gridy - pc_floory) * (pc_gridz - pc_floorz)
        weight1100 = (pc_gridt - pc_floort) * (pc_gridx - pc_floorx) * (pc_ceily - pc_gridy) * (pc_ceilz - pc_gridz)
        weight1101 = (pc_gridt - pc_floort) * (pc_gridx - pc_floorx) * (pc_ceily - pc_gridy) * (pc_gridz - pc_floorz)
        weight1110 = (pc_gridt - pc_floort) * (pc_gridx - pc_floorx) * (pc_gridy - pc_floory) * (pc_ceilz - pc_gridz)
        weight1111 = (pc_gridt - pc_floort) * (pc_gridx - pc_floorx) * (pc_gridy - pc_floory) * (pc_gridz - pc_floorz)

        all_weights = torch.cat([
            weight0000, weight0001, weight0010, weight0011, weight0100, weight0101, weight0110, weight0111,
            weight1000, weight1001, weight1010, weight1011, weight1100, weight1101, weight1110, weight1111
        ], 1).transpose(1, 0).contiguous() # [16, N]


        all_weights /= torch.prod(scale)

        all_weights[idx_query == -1] = 0
        all_weights /= all_weights.sum(0) + 1e-8

    return all_weights


class DevoxelizationGPU(Function):
    @staticmethod
    def forward(ctx, feat, indices, weights):
        out = torchsparse_cuda.deterministic_devoxelize_forward(feat.contiguous(), indices.contiguous().int(), weights.contiguous())
        ctx.for_backwards = (indices.contiguous().int(), weights, feat.shape[0])
        #print(f'Devox: {feat.shape}, {indices.shape}, {weights.shape} -> {out.shape}')

        return out

    @staticmethod
    def backward(ctx, grad_out):
        indices, weights, N = ctx.for_backwards
        grad_features = torchsparse_cuda.deterministic_devoxelize_backward(grad_out.contiguous(), indices, weights, N)
        #print(f'DevoxBack: {grad_out.shape}, {indices.shape}, {weights.shape} -> {out.shape}')

        return grad_features, None, None


devoxelize = DevoxelizationGPU.apply


def spdevoxelize(feat, indices, weights):
    return devoxelize(feat, indices, weights)
