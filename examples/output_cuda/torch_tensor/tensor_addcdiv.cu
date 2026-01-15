// PTO Program: tensor_addcdiv
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
__device__ float tensor1[8][8];
__device__ float tensor2[8][8];
__device__ float quot[8][8];
__device__ float scaled[8][8];
__device__ float result[8][8];

__global__ void tensor_addcdiv_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): self=TLOAD(...); tensor1=TLOAD(...); tensor2=TLOAD(...); quot=TDIV(...); scaled=TMULS(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
        tensor1[_row][_col] = input_t1[_row * 8 + _col];
        tensor2[_row][_col] = input_t2[_row * 8 + _col];
        quot[_row][_col] = tensor1[_row][_col] / tensor2[_row][_col];
        scaled[_row][_col] = quot[_row][_col] * 1.0f;
        result[_row][_col] = self[_row][_col] + scaled[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_addcdiv() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_addcdiv_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}