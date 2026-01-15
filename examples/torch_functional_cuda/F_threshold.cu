// PTO Program: F_threshold
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
__device__ float thresh_tile[8][8];
__device__ float value_tile[8][8];
__device__ float result[8][8];

__global__ void F_threshold_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): x=TLOAD(...); thresh_tile=TEXPANDS(...); value_tile=TEXPANDS(...); result=TMAX(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        thresh_tile[_row][_col] = 0.0f;
        value_tile[_row][_col] = 0.0f;
        result[_row][_col] = fmaxf(x[_row][_col], thresh_tile[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_threshold() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_threshold_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}