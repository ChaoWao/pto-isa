// PTO Program: F_logsigmoid
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
__device__ float neg_x[8][8];
__device__ float exp_neg_x[8][8];
__device__ float one_plus[8][8];
__device__ float softplus[8][8];
__device__ float result[8][8];

__global__ void F_logsigmoid_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); one_plus=TADDS(...); softplus=TLOG(...); result=TNEG(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        one_plus[_row][_col] = exp_neg_x[_row][_col] + 1.0f;
        softplus[_row][_col] = __logf(one_plus[_row][_col]);
        result[_row][_col] = -softplus[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_logsigmoid() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_logsigmoid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}