// PTO Program: tensor_mv
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
__device__ float vec[8][1];
__device__ float result[8][1];

__global__ void tensor_mv_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
    }

    // FUSED (1 ops): vec=TLOAD(...)
    if (_row < 8 && _col < 1) {
        vec[_row][_col] = input_vec[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_mv() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_mv_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}