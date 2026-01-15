// PTO Program: prims_pow
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

__device__ float base[8][8];
__device__ float exp[8][8];
__device__ float log_base[8][8];
__device__ float product[8][8];
__device__ float result[8][8];

__global__ void prims_pow_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (6 ops): base=TLOAD(...); exp=TLOAD(...); log_base=TLOG(...); product=TMUL(...); result=TEXP(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        base[_row][_col] = input_base[_row * 8 + _col];
        exp[_row][_col] = input_exp[_row * 8 + _col];
        log_base[_row][_col] = __logf(base[_row][_col]);
        product[_row][_col] = exp[_row][_col] * log_base[_row][_col];
        result[_row][_col] = __expf(product[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void prims_pow() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_pow_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}