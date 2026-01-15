// PTO Program: create_causal_mask
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

__device__ float mask[8][8];
__device__ float ones[8][8];

__global__ void create_causal_mask_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): mask=TEXPANDS(...); ones=TEXPANDS(...); mask_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        mask[_row][_col] = -1000000000.0f;
        ones[_row][_col] = 0.0f;
        mask_mem[_row * 8 + _col] = mask[_row][_col];
    }

}

void create_causal_mask() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    create_causal_mask_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}