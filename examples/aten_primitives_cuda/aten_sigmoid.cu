// PTO Program: aten_sigmoid
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
__device__ float t1[8][8];
__device__ float t2[8][8];
__device__ float t3[8][8];
__device__ float result[8][8];

__global__ void aten_sigmoid_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (6 ops): x=TLOAD(...); t1=TNEG(...); t2=TEXP(...); t3=TADDS(...); result=TRECIP(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        t1[_row][_col] = -x[_row][_col];
        t2[_row][_col] = __expf(t1[_row][_col]);
        t3[_row][_col] = t2[_row][_col] + 1.0f;
        result[_row][_col] = 1.0f / t3[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void aten_sigmoid() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_sigmoid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}