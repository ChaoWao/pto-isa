// PTO Program: F_mish
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
__device__ float one_plus_exp[8][8];
__device__ float softplus[8][8];
__device__ float sp_2[8][8];
__device__ float exp_2sp[8][8];
__device__ float tanh_num[8][8];
__device__ float tanh_den[8][8];
__device__ float tanh_out[8][8];
__device__ float result[8][8];

__global__ void F_mish_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 10 loop overheads saved

    // FUSED (11 ops): x=TLOAD(...); exp_x=TEXP(...); one_plus_exp=TADDS(...); softplus=TLOG(...); sp_2=TMULS(...); exp_2sp=TEXP(...); tanh_num=TADDS(...); tanh_den=TADDS(...); tanh_out=TDIV(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        one_plus_exp[_row][_col] = exp_x[_row][_col] + 1.0f;
        softplus[_row][_col] = __logf(one_plus_exp[_row][_col]);
        sp_2[_row][_col] = softplus[_row][_col] * 2.0f;
        exp_2sp[_row][_col] = __expf(sp_2[_row][_col]);
        tanh_num[_row][_col] = exp_2sp[_row][_col] + -1.0f;
        tanh_den[_row][_col] = exp_2sp[_row][_col] + 1.0f;
        tanh_out[_row][_col] = tanh_num[_row][_col] / tanh_den[_row][_col];
        result[_row][_col] = x[_row][_col] * tanh_out[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_mish() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_mish_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}