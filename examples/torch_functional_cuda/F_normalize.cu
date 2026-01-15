// PTO Program: F_normalize
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
__device__ float x_sq[8][8];
__device__ float row_sum[8][1];
__device__ float norm[8][1];
__device__ float result[8][8];

__global__ void F_normalize_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); x_sq=TMUL(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
    }

    // BARRIER: TROWSUM

    // FUSED (2 ops): norm=TSQRT(...); norm=TADDS(...)
    if (_row < 8 && _col < 1) {
        norm[_row][_col] = __fsqrt_rn(row_sum[_row][_col]);
        norm[_row][_col] = norm[_row][_col] + 1e-12f;
    }

    // BARRIER: TROWEXPANDDIV

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_normalize() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_normalize_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}