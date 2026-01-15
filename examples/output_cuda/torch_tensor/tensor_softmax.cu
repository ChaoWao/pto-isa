// PTO Program: tensor_softmax
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

__device__ float self[8][8];
__device__ float row_mean[8][1];
__device__ float shifted[8][8];
__device__ float exp_shifted[8][8];
__device__ float row_sum[8][1];
__device__ float result[8][8];

__global__ void tensor_softmax_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
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

    // BARRIER: TROWEXPANDDIV

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_softmax() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_softmax_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}