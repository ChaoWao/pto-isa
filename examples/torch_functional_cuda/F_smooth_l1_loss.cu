// PTO Program: F_smooth_l1_loss
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
__device__ float sq_diff[8][8];
__device__ float l2_part[8][8];
__device__ float l1_part[8][8];
__device__ float beta_tile[8][8];
__device__ float loss[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_smooth_l1_loss_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (8 ops): pred=TLOAD(...); target=TLOAD(...); diff=TSUB(...); abs_diff=TABS(...); sq_diff=TMUL(...); l2_part=TDIVS(...); l1_part=TADDS(...); loss=TMIN(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        diff[_row][_col] = pred[_row][_col] - target[_row][_col];
        abs_diff[_row][_col] = fabsf(diff[_row][_col]);
        sq_diff[_row][_col] = diff[_row][_col] * diff;
        l2_part[_row][_col] = sq_diff[_row][_col] / 2.0f;
        l1_part[_row][_col] = abs_diff[_row][_col] + -0.5f;
        loss[_row][_col] = fminf(l2_part[_row][_col], l1_part[_row][_col]);
    }

    // BARRIER: TROWSUM

    // BARRIER: TCOLSUM

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 64.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_smooth_l1_loss() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_smooth_l1_loss_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}