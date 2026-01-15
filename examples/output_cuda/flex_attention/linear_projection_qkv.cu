// PTO Program: linear_projection_qkv
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

__device__ float X[8][64];
__device__ float W_Q[64][8];
__device__ float W_K[64][8];
__device__ float W_V[64][8];
__device__ float Q[8][8];
__device__ float K[8][8];
__device__ float V[8][8];

__global__ void linear_projection_qkv_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (1 ops): X=TLOAD(...)
    if (_row < 8 && _col < 64) {
        X[_row][_col] = X_mem[_row * 8 + _col];
    }

    // FUSED (3 ops): W_Q=TLOAD(...); W_K=TLOAD(...); W_V=TLOAD(...)
    if (_row < 64 && _col < 8) {
        W_Q[_row][_col] = WQ_mem[_row * 8 + _col];
        W_K[_row][_col] = WK_mem[_row * 8 + _col];
        W_V[_row][_col] = WV_mem[_row * 8 + _col];
    }

    // BARRIER: TMATMUL

    // BARRIER: TMATMUL

    // BARRIER: TMATMUL

    // FUSED (3 ops): Q_mem=TSTORE(...); K_mem=TSTORE(...); V_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        Q_mem[_row * 8 + _col] = Q[_row][_col];
        K_mem[_row * 8 + _col] = K[_row][_col];
        V_mem[_row * 8 + _col] = V[_row][_col];
    }

}

void linear_projection_qkv() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    linear_projection_qkv_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}