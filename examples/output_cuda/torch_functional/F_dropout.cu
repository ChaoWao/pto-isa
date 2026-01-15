// PTO Program: F_dropout
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
__device__ float result[8][8];

__global__ void F_dropout_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); result=TMULS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        result[_row][_col] = x[_row][_col] * 1.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_dropout() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_dropout_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}