#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include "../common/gpu.cuh"


__global__ void deterministic_devoxelize_kernel(
    int M, 
    int c, 
    int l,
    const int *__restrict__ indices,    // [M, 8]
    const float *__restrict__ weight,   // [M, 8]
    const float *__restrict__ feat,     // [N, c]
    float *__restrict__ out             // [M, c]
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / c; // 0 ~ M
    int j = index % c; // 0 ~ c
    
    if(i < M){
        const int * indices_ = indices + l * i;
        const float * weight_ = weight + l * i;
        const float * feat_ = feat + j;
        float cur_feat;
        //#pragma unroll
        for(int k = 0; k < l; k++){
            cur_feat = (indices_[k] >= 0) ? feat_[indices_[k] * c]  : 0; 
            out[i * c + j] += weight_[k] * cur_feat;
        }
    }
}

void deterministic_devoxelize_wrapper(int M, int c, int l, const int * indices, const float * weight, const float * feat, float * out){
    deterministic_devoxelize_kernel<<<M, c>>>(M, c, l, indices, weight, feat, out);
}


__global__ void deterministic_devoxelize_grad_kernel(
    int M, 
    int N, 
    int c, 
    int l,
    const int *__restrict__ indices,     // [M, 8]
    const float *__restrict__ weight,    // [M, 8]
    const float *__restrict__ top_grad,  // [M, c]
    int *__restrict__ bottom_grad        // [N, c]
){    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    
    if(i < M){
        const int* indices_ = indices + l * i;
        const float *weight_ = weight + l * i;
        float cur_top_grad = top_grad[i * c + j];
        
        //#pragma unroll
        for(int k = 0; k < l; k++){
            float grad_float = weight_[k] * cur_top_grad;
            int64_t grad_int = (int64_t)round(grad_float * 1e10); // a hack, later it will / 1e10
            if (indices_[k] >= 0) 
                atomicAdd(&bottom_grad[indices_[k] * c + j], (int)grad_int);
        }
    }
}

void deterministic_devoxelize_grad_wrapper(int M, int N, int c, int l, const int *indices, const float * weight, const float * top_grad, int * bottom_grad){
    deterministic_devoxelize_grad_kernel<<<M, c>>>(M, N, c, l, indices, weight, top_grad, bottom_grad);
}
