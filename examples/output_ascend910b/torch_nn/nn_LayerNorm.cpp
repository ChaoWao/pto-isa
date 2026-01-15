// PTO Program: nn_LayerNorm
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class nn_LayerNormKernel {
public:
    __aicore__ inline nn_LayerNormKernel() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 8 * 8 * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, 8 * 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 64);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 2 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // BARRIER: TROWSUM
        // TROWSUM: ReduceSum along rows

        // FUSED (1 ops): TDIVS
        Divs(mean, row_sum, 8.0f, 64);

        // BARRIER: TROWEXPANDSUB
        // TROWEXPANDSUB: Barrier operation

        // FUSED (1 ops): TMUL
        Mul(squared, x_minus_mean, x_minus_mean, 64);

        // BARRIER: TROWSUM
        // TROWSUM: ReduceSum along rows

        // FUSED (3 ops): TDIVS; TADDS; TSQRT
        Divs(variance, var_sum, 8.0f, 64);
        Adds(var_eps, variance, 1e-05f, 64);
        Sqrt(std, var_eps, 64);

        // BARRIER: TROWEXPANDDIV
        // TROWEXPANDDIV: Barrier operation

        // FUSED (1 ops): TSTORE
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 64);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
};

extern "C" __global__ __aicore__ void nn_LayerNorm_kernel(GM_ADDR input, GM_ADDR output) {
    nn_LayerNormKernel op;
    op.Init(input, output);
    op.Process();
}