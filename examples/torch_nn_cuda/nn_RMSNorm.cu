// PTO Program: nn_RMSNorm
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
__device__ float x_squared[8][8];
__device__ float mean_sq_sum[8][1];
__device__ float mean_sq[8][1];
__device__ float mean_sq_eps[8][1];
__device__ float rms[8][1];
__device__ float result[8][8];

__global__ void nn_RMSNorm_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); x_squared=TMUL(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // FUSED (3 ops): mean_sq=TDIVS(...); mean_sq_eps=TADDS(...); rms=TSQRT(...)
    if (_row < 8 && _col < 1) {
        mean_sq[_row][_col] = mean_sq_sum[_row][_col] / 8.0f;
        mean_sq_eps[_row][_col] = mean_sq[_row][_col] + 1e-05f;
        rms[_row][_col] = __fsqrt_rn(mean_sq_eps[_row][_col]);
    }

    // FUSED (2 ops): result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = x[_row][_col] / rms[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_RMSNorm() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_RMSNorm_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}