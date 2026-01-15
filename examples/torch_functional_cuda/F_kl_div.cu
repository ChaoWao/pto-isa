// PTO Program: F_kl_div
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

__device__ float log_pred[8][8];
__device__ float target[8][8];
__device__ float log_target[8][8];
__device__ float diff[8][8];
__device__ float kl[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_kl_div_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (5 ops): log_pred=TLOAD(...); target=TLOAD(...); log_target=TLOG(...); diff=TSUB(...); kl=TMUL(...)
    if (_row < 8 && _col < 8) {
        log_pred[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        log_target[_row][_col] = __logf(target[_row][_col]);
        diff[_row][_col] = log_target[_row][_col] - log_pred[_row][_col];
        kl[_row][_col] = target[_row][_col] * diff;
    }

    // BARRIER: TROWSUM

    // BARRIER: TCOLSUM

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 64.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_kl_div() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_kl_div_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}