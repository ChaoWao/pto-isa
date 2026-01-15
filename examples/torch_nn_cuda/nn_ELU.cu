// PTO Program: nn_ELU
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
__device__ float pos_part[8][8];
__device__ float exp_x[8][8];
__device__ float exp_minus_1[8][8];
__device__ float scaled[8][8];
__device__ float neg_x[8][8];
__device__ float neg_relu[8][8];
__device__ float neg_part[8][8];
__device__ float neg_contrib[8][8];
__device__ float result[8][8];

__global__ void nn_ELU_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 9 loop overheads saved

    // FUSED (10 ops): x=TLOAD(...); pos_part=TRELU(...); exp_x=TEXP(...); exp_minus_1=TADDS(...); scaled=TMULS(...); neg_x=TNEG(...); neg_relu=TRELU(...); neg_contrib=TNEG(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        exp_x[_row][_col] = __expf(x[_row][_col]);
        exp_minus_1[_row][_col] = exp_x[_row][_col] + -1.0f;
        scaled[_row][_col] = exp_minus_1[_row][_col] * 1.0f;
        neg_x[_row][_col] = -scaled[_row][_col];
        neg_relu[_row][_col] = fmaxf(neg_x[_row][_col], 0.0f);
        neg_contrib[_row][_col] = -neg_relu[_row][_col];
        result[_row][_col] = pos_part[_row][_col] + neg_contrib[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_ELU() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_ELU_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}