// PTO Program: nn_GELU
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
__device__ float scaled_x[8][8];
__device__ float neg_scaled[8][8];
__device__ float exp_neg[8][8];
__device__ float one_plus[8][8];
__device__ float sigmoid_out[8][8];
__device__ float result[8][8];

__global__ void nn_GELU_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); scaled_x=TMULS(...); neg_scaled=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        scaled_x[_row][_col] = x[_row][_col] * 1.702f;
        neg_scaled[_row][_col] = -scaled_x[_row][_col];
        exp_neg[_row][_col] = __expf(neg_scaled[_row][_col]);
        one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
        sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
        result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_GELU() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_GELU_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}