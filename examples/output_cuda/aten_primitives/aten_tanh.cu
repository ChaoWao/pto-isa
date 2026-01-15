// PTO Program: aten_tanh
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
__device__ float exp_x[8][8];
__device__ float exp_neg_x[8][8];
__device__ float neg_x[8][8];
__device__ float numerator[8][8];
__device__ float denominator[8][8];
__device__ float result[8][8];

__global__ void aten_tanh_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); exp_x=TEXP(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); numerator=TSUB(...); denominator=TADD(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
        denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
        result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void aten_tanh() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_tanh_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}