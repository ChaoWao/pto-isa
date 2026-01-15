// PTO Program: F_hardsigmoid
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
__device__ float x_plus_3[8][8];
__device__ float scaled[8][8];
__device__ float zeros[8][8];
__device__ float ones[8][8];
__device__ float clamp_low[8][8];
__device__ float result[8][8];

__global__ void F_hardsigmoid_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); x_plus_3=TADDS(...); scaled=TDIVS(...); zeros=TEXPANDS(...); ones=TEXPANDS(...); clamp_low=TMAX(...); result=TMIN(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_plus_3[_row][_col] = x[_row][_col] + 3.0f;
        scaled[_row][_col] = x_plus_3[_row][_col] / 6.0f;
        zeros[_row][_col] = 0.0f;
        ones[_row][_col] = 1.0f;
        clamp_low[_row][_col] = fmaxf(scaled[_row][_col], zeros[_row][_col]);
        result[_row][_col] = fminf(clamp_low[_row][_col], ones[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_hardsigmoid() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_hardsigmoid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}