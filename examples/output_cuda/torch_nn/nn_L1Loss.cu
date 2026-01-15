// PTO Program: nn_L1Loss
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

__device__ float pred[8][8];
__device__ float target[8][8];
__device__ float diff[8][8];
__device__ float abs_diff[8][8];
__device__ float row_sum[8][1];
__device__ float total_sum[1][1];
__device__ float result[1][1];

__global__ void nn_L1Loss_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (4 ops): pred=TLOAD(...); target=TLOAD(...); diff=TSUB(...); abs_diff=TABS(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = pred_mem[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        diff[_row][_col] = pred[_row][_col] - target[_row][_col];
        abs_diff[_row][_col] = fabsf(diff[_row][_col]);
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // BARRIER: TCOLSUM
    // TCOLSUM: Requires block reduction - not shown in simplified example

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = total_sum[_row][_col] / 64.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_L1Loss() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_L1Loss_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}