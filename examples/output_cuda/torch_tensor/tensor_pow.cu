// PTO Program: tensor_pow
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
__device__ float log_self[8][8];
__device__ float scaled[8][8];
__device__ float result[8][8];

__global__ void tensor_pow_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): self=TLOAD(...); log_self=TLOG(...); scaled=TMULS(...); result=TEXP(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        log_self[_row][_col] = __logf(self[_row][_col]);
        scaled[_row][_col] = log_self[_row][_col] * 0.5f;
        result[_row][_col] = __expf(scaled[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_pow() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_pow_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}