// PTO Program: tensor_std
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
__device__ float row_sum[8][1];
__device__ float total[1][1];
__device__ float mean_val[8][8];
__device__ float centered[8][8];
__device__ float sq_centered[8][8];
__device__ float sq_row_sum[8][1];
__device__ float var_total[1][1];
__device__ float var[1][1];
__device__ float result[1][1];

__global__ void tensor_std_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
    }

    // BARRIER: TROWSUM

    // BARRIER: TCOLSUM

    // FUSED (1 ops): total=TDIVS(...)
    if (_row < 1 && _col < 1) {
        total[_row][_col] = total[_row][_col] / 64.0f;
    }

    // FUSED (3 ops): mean_val=TEXPANDS(...); centered=TSUB(...); sq_centered=TMUL(...)
    if (_row < 8 && _col < 8) {
        mean_val[_row][_col] = 0.0f;
        centered[_row][_col] = self[_row][_col] - mean_val[_row][_col];
        sq_centered[_row][_col] = centered[_row][_col] * centered[_row][_col];
    }

    // BARRIER: TROWSUM

    // BARRIER: TCOLSUM

    // FUSED (3 ops): var=TDIVS(...); result=TSQRT(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        var[_row][_col] = var_total[_row][_col] / 64.0f;
        result[_row][_col] = __fsqrt_rn(var[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_std() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_std_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}