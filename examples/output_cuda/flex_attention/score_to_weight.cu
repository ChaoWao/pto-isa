// PTO Program: score_to_weight
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

__device__ float scores[8][8];
__device__ float row_sum[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float weights[8][8];

__global__ void score_to_weight_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): scores=TLOAD(...)
    if (_row < 8 && _col < 8) {
        scores[_row][_col] = scores_mem[_row * 8 + _col];
    }

    // BARRIER: TROWSUM

    // FUSED (1 ops): row_sum=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // BARRIER: TROWEXPANDSUB

    // FUSED (1 ops): exp_scores=TEXP(...)
    if (_row < 8 && _col < 8) {
        exp_scores[_row][_col] = __expf(shifted[_row][_col]);
    }

    // BARRIER: TROWSUM

    // BARRIER: TROWEXPANDDIV

    // FUSED (1 ops): weights_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        weights_mem[_row * 8 + _col] = weights[_row][_col];
    }

}

void score_to_weight() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    score_to_weight_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}