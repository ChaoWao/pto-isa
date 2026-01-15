// PTO Program: nn_Softmax
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
__device__ float exp_x[8][8];
__device__ float sum_exp[8][1];
__device__ float result[8][8];

__global__ void nn_Softmax_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); exp_x=TEXP(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // FUSED (2 ops): result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = exp_x[_row][_col] / sum_exp[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Softmax() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Softmax_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}