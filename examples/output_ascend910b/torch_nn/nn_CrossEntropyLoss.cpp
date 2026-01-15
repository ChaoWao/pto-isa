// PTO Program: nn_CrossEntropyLoss
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class nn_CrossEntropyLossKernel {
public:
    __aicore__ inline nn_CrossEntropyLossKernel() {}
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

        // FUSED (3 ops): TLOAD; TLOAD; TEXP
        // TLOAD: Operation
        // TLOAD: Operation
        Exp(exp_pred, pred, 64);

        // BARRIER: TROWSUM
        // TROWSUM: ReduceSum along rows

        // FUSED (1 ops): TLOG
        Ln(log_sum, sum_exp, 64);

        // BARRIER: TROWEXPANDSUB
        // TROWEXPANDSUB: Barrier operation

        // FUSED (2 ops): TMUL; TNEG
        Mul(weighted, target, log_softmax, 64);
        Neg(neg_weighted, weighted, 64);

        // BARRIER: TROWSUM
        // TROWSUM: ReduceSum along rows

        // BARRIER: TCOLSUM
        // TCOLSUM: ReduceSum along columns

        // FUSED (2 ops): TDIVS; TSTORE
        Divs(result, total_sum, 8.0f, 64);
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

extern "C" __global__ __aicore__ void nn_CrossEntropyLoss_kernel(GM_ADDR input, GM_ADDR output) {
    nn_CrossEntropyLossKernel op;
    op.Init(input, output);
    op.Process();
}