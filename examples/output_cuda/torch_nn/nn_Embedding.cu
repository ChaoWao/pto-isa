// PTO Program: nn_Embedding
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

__device__ float indices_onehot[8][64];
__device__ float weight[64][8];
__device__ float result[8][8];

__global__ void nn_Embedding_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): indices_onehot=TLOAD(...)
    if (_row < 8 && _col < 64) {
        indices_onehot[_row][_col] = indices_mem[_row * 8 + _col];
    }

    // FUSED (1 ops): weight=TLOAD(...)
    if (_row < 64 && _col < 8) {
        weight[_row][_col] = weight_mem[_row * 8 + _col];
    }

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Embedding() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Embedding_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}