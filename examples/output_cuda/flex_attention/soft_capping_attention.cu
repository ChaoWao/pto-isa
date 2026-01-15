// PTO Program: soft_capping_attention
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
__device__ float scores[8][8];
__device__ float scaled[8][8];
__device__ float x_div_cap[8][8];
__device__ float two_x[8][8];
__device__ float exp_2x[8][8];
__device__ float exp_minus_1[8][8];
__device__ float exp_plus_1[8][8];
__device__ float tanh_x[8][8];
__device__ float capped_scores[8][8];
__device__ float row_sum[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float attn[8][8];
__device__ float output[8][8];

__global__ void soft_capping_attention_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 9 loop overheads saved

    // FUSED (3 ops): Q=TLOAD(...); K=TLOAD(...); V=TLOAD(...)
    if (_row < 8 && _col < 8) {
        Q[_row][_col] = Q_mem[_row * 8 + _col];
        K[_row][_col] = K_mem[_row * 8 + _col];
        V[_row][_col] = V_mem[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (8 ops): scaled=TMULS(...); x_div_cap=TDIVS(...); two_x=TMULS(...); exp_2x=TEXP(...); exp_minus_1=TADDS(...); exp_plus_1=TADDS(...); tanh_x=TDIV(...); capped_scores=TMULS(...)
    if (_row < 8 && _col < 8) {
        scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
        x_div_cap[_row][_col] = scaled[_row][_col] / 50.0f;
        two_x[_row][_col] = x_div_cap[_row][_col] * 2.0f;
        exp_2x[_row][_col] = __expf(two_x[_row][_col]);
        exp_minus_1[_row][_col] = exp_2x[_row][_col] + -1.0f;
        exp_plus_1[_row][_col] = exp_2x[_row][_col] + 1.0f;
        tanh_x[_row][_col] = exp_minus_1[_row][_col] / exp_plus_1[_row][_col];
        capped_scores[_row][_col] = tanh_x[_row][_col] * 50.0f;
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

    // BARRIER: TMATMUL

    // FUSED (1 ops): output_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void soft_capping_attention() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    soft_capping_attention_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}