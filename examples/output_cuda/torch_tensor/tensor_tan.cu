// PTO Program: tensor_tan
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
__device__ float x4[8][8];
__device__ float x5[8][8];
__device__ float sin_t1[8][8];
__device__ float sin_t2[8][8];
__device__ float sin_temp[8][8];
__device__ float sin_val[8][8];
__device__ float cos_t1[8][8];
__device__ float cos_t2[8][8];
__device__ float ones[8][8];
__device__ float cos_temp[8][8];
__device__ float cos_val[8][8];
__device__ float result[8][8];

__global__ void tensor_tan_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 15 loop overheads saved

    // FUSED (16 ops): x=TLOAD(...); x2=TMUL(...); x3=TMUL(...); x4=TMUL(...); x5=TMUL(...); sin_t1=TDIVS(...); sin_t2=TDIVS(...); sin_temp=TSUB(...); sin_val=TADD(...); cos_t1=TDIVS(...); cos_t2=TDIVS(...); ones=TEXPANDS(...); cos_temp=TSUB(...); cos_val=TADD(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x2[_row][_col] = x[_row][_col] * x[_row][_col];
        x3[_row][_col] = x2[_row][_col] * x[_row][_col];
        x4[_row][_col] = x2[_row][_col] * x2[_row][_col];
        x5[_row][_col] = x3[_row][_col] * x2[_row][_col];
        sin_t1[_row][_col] = x3[_row][_col] / 6.0f;
        sin_t2[_row][_col] = x5[_row][_col] / 120.0f;
        sin_temp[_row][_col] = x[_row][_col] - sin_t1[_row][_col];
        sin_val[_row][_col] = sin_temp[_row][_col] + sin_t2[_row][_col];
        cos_t1[_row][_col] = x2[_row][_col] / 2.0f;
        cos_t2[_row][_col] = x4[_row][_col] / 24.0f;
        ones[_row][_col] = 1.0f;
        cos_temp[_row][_col] = ones[_row][_col] - cos_t1[_row][_col];
        cos_val[_row][_col] = cos_temp[_row][_col] + cos_t2[_row][_col];
        result[_row][_col] = sin_val[_row][_col] / cos_val[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_tan() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_tan_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}