// PTO Program: nn_SmoothL1Loss
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class nn_SmoothL1LossKernel {
public:
    __aicore__ inline nn_SmoothL1LossKernel() {}
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

        // Loop fusion: 8 loop overheads saved

        // FUSED (8 ops): TLOAD; TLOAD; TSUB; TABS; TMUL; TDIVS; TADDS; TMIN
        // TLOAD: Operation
        // TLOAD: Operation
        Sub(diff, pred, target, 64);
        Abs(abs_diff, diff, 64);
        Mul(squared, diff, diff, 64);
        Divs(l2_term, squared, 2.0f, 64);
        Adds(l1_term, abs_diff, -0.5f, 64);
        Min(smooth, l2_term, l1_term, 64);

        // BARRIER: TROWSUM
        // TROWSUM: ReduceSum along rows

        // BARRIER: TCOLSUM
        // TCOLSUM: ReduceSum along columns

        // FUSED (2 ops): TDIVS; TSTORE
        Divs(result, total_sum, 64.0f, 64);
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

extern "C" __global__ __aicore__ void nn_SmoothL1Loss_kernel(GM_ADDR input, GM_ADDR output) {
    nn_SmoothL1LossKernel op;
    op.Init(input, output);
    op.Process();
}