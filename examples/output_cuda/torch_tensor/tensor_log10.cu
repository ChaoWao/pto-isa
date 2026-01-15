// PTO Program: tensor_log10
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
__device__ float ln_val[8][8];
__device__ float result[8][8];

__global__ void tensor_log10_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): self=TLOAD(...); ln_val=TLOG(...); result=TMULS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        ln_val[_row][_col] = __logf(self[_row][_col]);
        result[_row][_col] = ln_val[_row][_col] * 0.4342944819032518f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_log10() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_log10_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}