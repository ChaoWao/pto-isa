// PTO Program: nn_CrossEntropyLoss
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
__device__ float exp_pred[8][8];
__device__ float sum_exp[8][1];
__device__ float log_sum[8][1];
__device__ float log_softmax[8][8];
__device__ float weighted[8][8];
__device__ float neg_weighted[8][8];
__device__ float row_sum[8][1];
__device__ float total_sum[1][1];
__device__ float result[1][1];

__global__ void nn_CrossEntropyLoss_kernel() {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (3 ops): pred=TLOAD(...); target=TLOAD(...); exp_pred=TEXP(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = pred_mem[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        exp_pred[_row][_col] = __expf(pred[_row][_col]);
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // FUSED (1 ops): log_sum=TLOG(...)
    if (_row < 8 && _col < 1) {
        log_sum[_row][_col] = __logf(sum_exp[_row][_col]);
    }

    // BARRIER: TROWEXPANDSUB
    // TROWEXPANDSUB: Barrier operation

    // FUSED (2 ops): weighted=TMUL(...); neg_weighted=TNEG(...)
    if (_row < 8 && _col < 8) {
        weighted[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
        neg_weighted[_row][_col] = -weighted[_row][_col];
    }

    // BARRIER: TROWSUM
    // TROWSUM: Requires warp reduction - not shown in simplified example

    // BARRIER: TCOLSUM
    // TCOLSUM: Requires block reduction - not shown in simplified example

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = total_sum[_row][_col] / 8.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_CrossEntropyLoss() {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_CrossEntropyLoss_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}