// PTO Program: F_nll_loss
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

__device__ float log_probs[8][8];
__device__ float target[8][8];
__device__ float weighted[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_nll_loss_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (3 ops): log_probs=TLOAD(...); target=TLOAD(...); weighted=TMUL(...)
    if (_row < 8 && _col < 8) {
        log_probs[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        weighted[_row][_col] = target[_row][_col] * log_probs[_row][_col];
    }

    // BARRIER: TROWSUM

    // BARRIER: TCOLSUM

    // FUSED (3 ops): result=TNEG(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = -result[_row][_col];
        result[_row][_col] = result[_row][_col] / 8.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_nll_loss() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_nll_loss_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}