// PTO Program: F_relu6
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
__device__ float relu_out[8][8];
__device__ float six[8][8];
__device__ float result[8][8];

__global__ void F_relu6_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): x=TLOAD(...); relu_out=TRELU(...); six=TEXPANDS(...); result=TMIN(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        relu_out[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        six[_row][_col] = 6.0f;
        result[_row][_col] = fminf(relu_out[_row][_col], six[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_relu6() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_relu6_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}