// PTO Program: output_projection
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

__device__ float attn_out[8][8];
__device__ float W_O[8][64];
__device__ float output[8][64];

__global__ void output_projection_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): attn_out=TLOAD(...)
    if (_row < 8 && _col < 8) {
        attn_out[_row][_col] = attn_mem[_row * 8 + _col];
    }

    // FUSED (1 ops): W_O=TLOAD(...)
    if (_row < 8 && _col < 64) {
        W_O[_row][_col] = WO_mem[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // FUSED (1 ops): output_mem=TSTORE(...)
    if (_row < 8 && _col < 64) {
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void output_projection() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    output_projection_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}