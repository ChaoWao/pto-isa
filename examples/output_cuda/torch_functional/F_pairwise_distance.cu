// PTO Program: F_pairwise_distance
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
__device__ float diff[8][8];
__device__ float sq_diff[8][8];
__device__ float row_sum[8][1];
__device__ float result[8][1];

__global__ void F_pairwise_distance_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (4 ops): x1=TLOAD(...); x2=TLOAD(...); diff=TSUB(...); sq_diff=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        diff[_row][_col] = x1[_row][_col] - x2[_row][_col];
        sq_diff[_row][_col] = diff[_row][_col] * diff;
    }

    // BARRIER: TROWSUM

    // FUSED (2 ops): result=TSQRT(...); output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        result[_row][_col] = __fsqrt_rn(row_sum[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_pairwise_distance() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_pairwise_distance_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}