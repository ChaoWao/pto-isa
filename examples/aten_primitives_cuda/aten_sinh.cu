// PTO Program: aten_sinh
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
__device__ float x_squared[8][8];
__device__ float term[8][8];
__device__ float result[8][8];

__global__ void aten_sinh_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 16 loop overheads saved

    // FUSED (17 ops): x=TLOAD(...); result=TMULS(...); x_squared=TMUL(...); term=TMULS(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        result[_row][_col] = x[_row][_col] * 1.0f;
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
        term[_row][_col] = x[_row][_col] * 1.0f;
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 6.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 20.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 42.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 72.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void aten_sinh() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_sinh_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}