// PTO Program: tensor_clamp
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
__device__ float min_tile[8][8];
__device__ float max_tile[8][8];
__device__ float clamp_low[8][8];
__device__ float result[8][8];

__global__ void tensor_clamp_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (6 ops): self=TLOAD(...); min_tile=TEXPANDS(...); max_tile=TEXPANDS(...); clamp_low=TMAX(...); result=TMIN(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        min_tile[_row][_col] = -1.0f;
        max_tile[_row][_col] = 1.0f;
        clamp_low[_row][_col] = fmaxf(self[_row][_col], min_tile[_row][_col]);
        result[_row][_col] = fminf(clamp_low[_row][_col], max_tile[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_clamp() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_clamp_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}