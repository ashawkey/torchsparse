#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// FNV-1a hash: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function

//hashing
//input N*4 int32 tensor output N*1 int64 tensor
__global__ void hash_kernel(int N, int D, const int *__restrict__ data, long int *__restrict__ out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        data += i * D;
        unsigned long long hash = 14695981039346656037UL; // uint64
        for(int j = 0; j < D; j++){
            hash ^= (unsigned int)data[j]; // should not be a byte ???
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF); // ???
        out[i] = hash;
    }
}


void hash_wrapper(int N, int D, const int * data, long int * out){
    hash_kernel<<<ceil((double)N/512), 512>>>(N, D, data, out);
}

//kernel hashing: given data D and offset map K, generate D x K
//input N*4 int32 tensor, |K|*3 int32 tensor, output |K|*N int64 tensor
__global__ void kernel_hash_kernel(int N, int D, int K, const int *__restrict__ data, const int * __restrict__ kernel_offset, long int *__restrict__ out){
        
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int k = idx % K;
    int i = idx / K;
    int cur_coord[5]; // at most 4d conv
    
    if(i < N){
        data += i * D;
        for(int j = 0; j < D - 1; j++){
            cur_coord[j] = data[j]+kernel_offset[k*(D-1)+j];
        }
        cur_coord[D-1] = data[D-1];
        unsigned long long hash = 14695981039346656037UL;
        for(int j = 0; j < D; j++){
            hash ^= (unsigned int)cur_coord[j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[k*N+i] = hash;
    }
}


void kernel_hash_wrapper(int N, int D, int K, const int * data, const int *kernel_offset, long int * out){
    kernel_hash_kernel<<<ceil((double)(N*K)/512), 512, K*3*sizeof(int)>>>(N, D, K, data, kernel_offset, out);
}

////////////////////////////////////

/*
deformable kernel hashing
params:
    data: Int[N, 4]
    kernel_offset: Int[N, K, 3]
return:
    out: Long[N, K]
*/
__global__ void deform_kernel_hash_kernel(int N, int D, int K, const int *__restrict__ data, const int * __restrict__ kernel_offset, long int *__restrict__ out){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int k = idx % K;
    int i = idx / K;
    int cur_coord[5];

    if (i < N) {
        data += i * D;
        kernel_offset += i * K * (D-1); // the difference! each point has different deformations.
        for (int j = 0; j < D-1 ; j++) {
            cur_coord[j] = data[j] + kernel_offset[k*(D-1)+j];
        }
        cur_coord[D-1] = data[D-1];
        unsigned long long hash = 14695981039346656037UL;
        for(int j = 0; j < D; j++){
            hash ^= (unsigned int)cur_coord[j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[k*N+i] = hash;
    }
}


void deform_kernel_hash_wrapper(int N, int D, int K, const int * data, const int *kernel_offset, long int * out){
    deform_kernel_hash_kernel<<<ceil((double)(N*K)/512), 512, K*3*sizeof(int)>>>(N, D, K, data, kernel_offset, out);
}

