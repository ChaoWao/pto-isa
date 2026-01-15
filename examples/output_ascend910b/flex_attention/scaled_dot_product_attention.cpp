// PTO Program: scaled_dot_product_attention
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class scaled_dot_product_attentionKernel {
public:
    __aicore__ inline scaled_dot_product_attentionKernel() {}
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

        // Loop fusion: 2 loop overheads saved

        // FUSED (3 ops): TLOAD; TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation
        // TLOAD: Operation

        // BARRIER: TMATMUL

        // FUSED (1 ops): TMULS
        Muls(scaled_scores, scores, 0.35355339059327373f, 64);

        // BARRIER: TROWSUM

        // FUSED (1 ops): TDIVS
        Divs(row_max, row_max, 8.0f, 64);

        // BARRIER: TROWEXPANDSUB

        // FUSED (1 ops): TEXP
        Exp(exp_scores, shifted, 64);

        // BARRIER: TROWSUM

        // BARRIER: TROWEXPANDDIV

        // BARRIER: TMATMUL

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

extern "C" __global__ __aicore__ void scaled_dot_product_attention_kernel(GM_ADDR input, GM_ADDR output) {
    scaled_dot_product_attentionKernel op;
    op.Init(input, output);
    op.Process();
}