// PTO Program: F_cross_entropy
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

__device__ float logits[8][8];
__device__ float target[8][8];
__device__ float row_mean[8][1];
__device__ float shifted[8][8];
__device__ float exp_shifted[8][8];
__device__ float row_sum[8][1];
__device__ float log_sum[8][1];
__device__ float log_softmax[8][8];
__device__ float ce[8][8];
__device__ float ce_row[8][1];
__device__ float result[1][1];

__global__ void F_cross_entropy_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (2 ops): logits=TLOAD(...); target=TLOAD(...)
    if (_row < 8 && _col < 8) {
        logits[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
    }

    // BARRIER: TROWSUM

    // FUSED (1 ops): row_mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_mean[_row][_col] = row_mean[_row][_col] / 8.0f;
    }

    // BARRIER: TROWEXPANDSUB

    // FUSED (1 ops): exp_shifted=TEXP(...)
    if (_row < 8 && _col < 8) {
        exp_shifted[_row][_col] = __expf(shifted[_row][_col]);
    }

    // BARRIER: TROWSUM

    // FUSED (1 ops): log_sum=TLOG(...)
    if (_row < 8 && _col < 1) {
        log_sum[_row][_col] = __logf(row_sum[_row][_col]);
    }

    // BARRIER: TROWEXPANDSUB

    // FUSED (2 ops): ce=TMUL(...); ce=TNEG(...)
    if (_row < 8 && _col < 8) {
        ce[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
        ce[_row][_col] = -ce[_row][_col];
    }

    // BARRIER: TROWSUM

    // BARRIER: TCOLSUM

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 8.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_cross_entropy() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_cross_entropy_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}