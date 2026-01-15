// PTO Program: nn_LeakyReLU
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
__device__ float pos_part[8][8];
__device__ float neg_x[8][8];
__device__ float neg_relu[8][8];
__device__ float neg_part[8][8];
__device__ float scaled_neg[8][8];
__device__ float result[8][8];

__global__ void nn_LeakyReLU_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); pos_part=TRELU(...); neg_x=TNEG(...); neg_relu=TRELU(...); neg_part=TNEG(...); scaled_neg=TMULS(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        neg_x[_row][_col] = -x[_row][_col];
        neg_relu[_row][_col] = fmaxf(neg_x[_row][_col], 0.0f);
        neg_part[_row][_col] = -neg_relu[_row][_col];
        scaled_neg[_row][_col] = neg_part[_row][_col] * 0.01f;
        result[_row][_col] = pos_part[_row][_col] + scaled_neg[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_LeakyReLU() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_LeakyReLU_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}