// PTO Program: prims_log
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
__device__ float result[8][8];

__global__ void prims_log_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); result=TLOG(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        result[_row][_col] = __logf(x[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void prims_log() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_log_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}