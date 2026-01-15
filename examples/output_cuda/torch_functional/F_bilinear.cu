// PTO Program: F_bilinear
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

__device__ float x1[8][8];
__device__ float x2[8][8];
__device__ float weight[8][8];
__device__ float temp[8][8];
__device__ float output[8][8];

__global__ void F_bilinear_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (3 ops): x1=TLOAD(...); x2=TLOAD(...); weight=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        weight[_row][_col] = weight_mem[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (2 ops): output=TMUL(...); output_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row][_col] = temp[_row][_col] * x2[_row][_col];
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void F_bilinear() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_bilinear_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}