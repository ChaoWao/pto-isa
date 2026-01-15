// PTO Program: nn_LayerNorm
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
__device__ float row_sum[8][1];
__device__ float mean[8][1];
__device__ float x_minus_mean[8][8];
__device__ float squared[8][8];
__device__ float var_sum[8][1];
__device__ float variance[8][1];
__device__ float var_eps[8][1];
__device__ float std[8][1];
__device__ float result[8][8];

__global__ void nn_LayerNorm_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // FUSED (1 ops): mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        mean[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // BARRIER: TROWEXPANDSUB
    // TROWEXPANDSUB: Barrier operation

    // FUSED (1 ops): squared=TMUL(...)
    if (_row < 8 && _col < 8) {
        squared[_row][_col] = x_minus_mean[_row][_col] * x_minus_mean[_row][_col];
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // FUSED (3 ops): variance=TDIVS(...); var_eps=TADDS(...); std=TSQRT(...)
    if (_row < 8 && _col < 1) {
        variance[_row][_col] = var_sum[_row][_col] / 8.0f;
        var_eps[_row][_col] = variance[_row][_col] + 1e-05f;
        std[_row][_col] = __fsqrt_rn(var_eps[_row][_col]);
    }

    // BARRIER: TROWEXPANDDIV
    // TROWEXPANDDIV: Barrier operation

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_LayerNorm() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_LayerNorm_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}