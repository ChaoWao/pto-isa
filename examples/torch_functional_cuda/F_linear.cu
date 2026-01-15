// PTO Program: F_linear
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

__device__ float x[8][8];
__device__ float weight[8][8];
__device__ float output[8][8];
__device__ float bias[8][8];

__global__ void F_linear_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); weight=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        weight[_row][_col] = weight_mem[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (3 ops): bias=TLOAD(...); output=TADD(...); output_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        bias[_row][_col] = bias_mem[_row * 8 + _col];
        output[_row][_col] = output[_row][_col] + bias[_row][_col];
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void F_linear() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_linear_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}