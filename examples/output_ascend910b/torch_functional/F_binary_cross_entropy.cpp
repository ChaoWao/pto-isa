// PTO Program: F_binary_cross_entropy
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_binary_cross_entropyKernel {
public:
    __aicore__ inline F_binary_cross_entropyKernel() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 8 * 8 * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, 8 * 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
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

        // Loop fusion: 12 loop overheads saved

        // FUSED (12 ops): TLOAD; TLOAD; TLOG; TMULS; TADDS; TLOG; TMULS; TADDS; TMUL; TMUL; TADD; TNEG
        // TLOAD: Operation
        // TLOAD: Operation
        Ln(log_pred, pred, 64);
        Muls(one_minus_pred, pred, -1.0f, 64);
        Adds(one_minus_pred, one_minus_pred, 1.0f, 64);
        Ln(log_one_minus, one_minus_pred, 64);
        Muls(one_minus_target, target, -1.0f, 64);
        Adds(one_minus_target, one_minus_target, 1.0f, 64);
        Mul(term1, target, log_pred, 64);
        Mul(term2, one_minus_target, log_one_minus, 64);
        Add(bce, term1, term2, 64);
        Neg(bce, bce, 64);

        // BARRIER: TROWSUM

        // BARRIER: TCOLSUM

        // FUSED (2 ops): TDIVS; TSTORE
        Divs(result, result, 64.0f, 64);
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

extern "C" __global__ __aicore__ void F_binary_cross_entropy_kernel(GM_ADDR input, GM_ADDR output) {
    F_binary_cross_entropyKernel op;
    op.Init(input, output);
    op.Process();
}