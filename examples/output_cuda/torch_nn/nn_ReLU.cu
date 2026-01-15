// PTO Program: nn_ReLU
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

__global__ void nn_ReLU_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); result=TRELU(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        result[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_ReLU() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_ReLU_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}