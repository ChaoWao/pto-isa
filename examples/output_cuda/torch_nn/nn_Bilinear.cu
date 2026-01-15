// PTO Program: nn_Bilinear
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
__device__ float product[8][8];
__device__ float weight[8][8];
__device__ float result[8][8];

__global__ void nn_Bilinear_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): x1=TLOAD(...); x2=TLOAD(...); weight=TLOAD(...); product=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        weight[_row][_col] = weight_mem[_row * 8 + _col];
        product[_row][_col] = x1[_row][_col] * x2[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Bilinear() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Bilinear_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}