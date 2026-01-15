// PTO Program: tensor_logit
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
__device__ float one_minus[8][8];
__device__ float ratio[8][8];
__device__ float result[8][8];

__global__ void tensor_logit_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): self=TLOAD(...); one_minus=TMULS(...); one_minus=TADDS(...); one_minus=TADDS(...); ratio=TDIV(...); result=TLOG(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        one_minus[_row][_col] = self[_row][_col] * -1.0f;
        one_minus[_row][_col] = one_minus[_row][_col] + 1.0f;
        one_minus[_row][_col] = one_minus[_row][_col] + 1e-06f;
        ratio[_row][_col] = self[_row][_col] / one_minus[_row][_col];
        result[_row][_col] = __logf(ratio[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_logit() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_logit_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}