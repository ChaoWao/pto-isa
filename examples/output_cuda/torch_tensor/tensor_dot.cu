// PTO Program: tensor_dot
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

__device__ float self[1][64];
__device__ float other[1][64];
__device__ float prod[1][64];
__device__ float result[1][1];

__global__ void tensor_dot_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): self=TLOAD(...); other=TLOAD(...); prod=TMUL(...)
    if (_row < 1 && _col < 64) {
        self[_row][_col] = input_self[_row * 8 + _col];
        other[_row][_col] = input_other[_row * 8 + _col];
        prod[_row][_col] = self[_row][_col] * other[_row][_col];
    }

    // BARRIER: TROWSUM

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_dot() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_dot_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}