// PTO Program: prims_add
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

__device__ float a[8][8];
__device__ float b[8][8];
__device__ float result[8][8];

__global__ void prims_add_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): a=TLOAD(...); b=TLOAD(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        a[_row][_col] = input_a[_row * 8 + _col];
        b[_row][_col] = input_b[_row * 8 + _col];
        result[_row][_col] = a[_row][_col] + b[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void prims_add() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_add_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}