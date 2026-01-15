// PTO Program: F_cosine_similarity
// Auto-generated CUDA code from PTO ISA Compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ float x1[8][8];
__device__ float x2[8][8];
__device__ float dot_prod[8][8];
__device__ float x1_sq[8][8];
__device__ float x2_sq[8][8];
__device__ float dot_sum[8][1];
__device__ float x1_norm_sq[8][1];
__device__ float x2_norm_sq[8][1];
__device__ float x1_norm[8][1];
__device__ float x2_norm[8][1];
__device__ float norm_prod[8][1];
__device__ float result[8][1];

__global__ void F_cosine_similarity_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (3 ops): x1=TLOAD(...); x2=TLOAD(...); dot_prod=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        dot_prod[_row][_col] = x1[_row][_col] * x2[_row][_col];
    }

    // BARRIER: TROWSUM

    // FUSED (2 ops): x1_sq=TMUL(...); x2_sq=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1_sq[_row][_col] = x1[_row][_col] * x1[_row][_col];
        x2_sq[_row][_col] = x2[_row][_col] * x2[_row][_col];
    }

    // BARRIER: TROWSUM

    // BARRIER: TROWSUM

    // FUSED (6 ops): x1_norm=TSQRT(...); x2_norm=TSQRT(...); norm_prod=TMUL(...); norm_prod=TADDS(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        x1_norm[_row][_col] = __fsqrt_rn(x1_norm_sq[_row][_col]);
        x2_norm[_row][_col] = __fsqrt_rn(x2_norm_sq[_row][_col]);
        norm_prod[_row][_col] = x1_norm[_row][_col] * x2_norm[_row][_col];
        norm_prod[_row][_col] = norm_prod[_row][_col] + 1e-08f;
        result[_row][_col] = dot_sum[_row][_col] / norm_prod[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_cosine_similarity() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_cosine_similarity_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}