// PTO Program: tensor_mm
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
__device__ float mat2[8][8];
__device__ float result[8][8];

__global__ void tensor_mm_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (2 ops): self=TLOAD(...); mat2=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
        mat2[_row][_col] = input_mat2[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_mm() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_mm_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}