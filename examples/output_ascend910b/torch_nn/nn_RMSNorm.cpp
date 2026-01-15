// PTO Program: nn_RMSNorm
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class nn_RMSNormKernel {
public:
    __aicore__ inline nn_RMSNormKernel() {}
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

        // Loop fusion: 4 loop overheads saved

        // FUSED (2 ops): TLOAD; TMUL
        // TLOAD: Operation
        Mul(x_squared, x, x, 64);

        // BARRIER: TROWSUM
        // TROWSUM: ReduceSum along rows

        // FUSED (3 ops): TDIVS; TADDS; TSQRT
        Divs(mean_sq, mean_sq_sum, 8.0f, 64);
        Adds(mean_sq_eps, mean_sq, 1e-05f, 64);
        Sqrt(rms, mean_sq_eps, 64);

        // FUSED (2 ops): TDIV; TSTORE
        Div(result, x, rms, 64);
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

extern "C" __global__ __aicore__ void nn_RMSNorm_kernel(GM_ADDR input, GM_ADDR output) {
    nn_RMSNormKernel op;
    op.Init(input, output);
    op.Process();
}