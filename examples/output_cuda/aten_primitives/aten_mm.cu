// PTO Program: aten_mm
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

__device__ float a[8][8];
__device__ float b[8][8];
__device__ float result[8][8];

__global__ void aten_mm_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): a=TLOAD(...); b=TLOAD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        a[_row][_col] = input_a[_row * 8 + _col];
        b[_row][_col] = input_b[_row * 8 + _col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void aten_mm() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_mm_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}