// PTO Program: scaled_dot_product_attention
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

__device__ float Q[8][8];
__device__ float K[8][8];
__device__ float V[8][8];
__device__ float K_T[8][8];
__device__ float scores[8][8];
__device__ float scaled_scores[8][8];
__device__ float row_max[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float row_sum[8][1];
__device__ float attention_weights[8][8];
__device__ float output[8][8];

__global__ void scaled_dot_product_attention_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): Q=TLOAD(...); K=TLOAD(...); V=TLOAD(...)
    if (_row < 8 && _col < 8) {
        Q[_row][_col] = Q_mem[_row * 8 + _col];
        K[_row][_col] = K_mem[_row * 8 + _col];
        V[_row][_col] = V_mem[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (1 ops): scaled_scores=TMULS(...)
    if (_row < 8 && _col < 8) {
        scaled_scores[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
    }

    // BARRIER: TROWSUM

    // FUSED (1 ops): row_max=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_max[_row][_col] = row_max[_row][_col] / 8.0f;
    }

    // BARRIER: TROWEXPANDSUB

    // FUSED (1 ops): exp_scores=TEXP(...)
    if (_row < 8 && _col < 8) {
        exp_scores[_row][_col] = __expf(shifted[_row][_col]);
    }

    // BARRIER: TROWSUM

    // BARRIER: TROWEXPANDDIV

    // BARRIER: TMATMUL

    // FUSED (1 ops): output_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void scaled_dot_product_attention() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    scaled_dot_product_attention_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}