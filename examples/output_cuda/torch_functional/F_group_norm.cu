// PTO Program: F_group_norm
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

__device__ float x[8][8];
__device__ float mean[8][1];
__device__ float centered[8][8];
__device__ float sq_centered[8][8];
__device__ float var[8][1];
__device__ float std[8][1];
__device__ float result[8][8];

__global__ void F_group_norm_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // BARRIER: TROWSUM

    // FUSED (1 ops): mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        mean[_row][_col] = mean[_row][_col] / 8.0f;
    }

    // BARRIER: TROWEXPANDSUB

    // FUSED (1 ops): sq_centered=TMUL(...)
    if (_row < 8 && _col < 8) {
        sq_centered[_row][_col] = centered[_row][_col] * centered[_row][_col];
    }

    // BARRIER: TROWSUM

    // FUSED (3 ops): var=TDIVS(...); var=TADDS(...); std=TSQRT(...)
    if (_row < 8 && _col < 1) {
        var[_row][_col] = var[_row][_col] / 8.0f;
        var[_row][_col] = var[_row][_col] + 1e-05f;
        std[_row][_col] = __fsqrt_rn(var[_row][_col]);
    }

    // BARRIER: TROWEXPANDDIV

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_group_norm() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_group_norm_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}