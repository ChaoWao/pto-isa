// PTO Program: tensor_sin
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
__device__ float x2[8][8];
__device__ float x3[8][8];
__device__ float x5[8][8];
__device__ float term1[8][8];
__device__ float term2[8][8];
__device__ float temp[8][8];
__device__ float result[8][8];

__global__ void tensor_sin_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (9 ops): x=TLOAD(...); x2=TMUL(...); x3=TMUL(...); x5=TMUL(...); term1=TDIVS(...); term2=TDIVS(...); temp=TSUB(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x2[_row][_col] = x[_row][_col] * x[_row][_col];
        x3[_row][_col] = x2[_row][_col] * x[_row][_col];
        x5[_row][_col] = x3[_row][_col] * x2[_row][_col];
        term1[_row][_col] = x3[_row][_col] / 6.0f;
        term2[_row][_col] = x5[_row][_col] / 120.0f;
        temp[_row][_col] = x[_row][_col] - term1[_row][_col];
        result[_row][_col] = temp[_row][_col] + term2[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_sin() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_sin_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}