// PTO Program: aten_mean_row
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
__device__ float sum_result[8][1];
__device__ float result[8][1];

__global__ void aten_mean_row_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        result[_row][_col] = sum_result[_row][_col] / 8.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void aten_mean_row() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_mean_row_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}