// PTO Program: F_smooth_l1_loss
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_smooth_l1_lossKernel {
public:
    __aicore__ inline F_smooth_l1_lossKernel() {}
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

        // Loop fusion: 8 loop overheads saved

        // FUSED (8 ops): TLOAD; TLOAD; TSUB; TABS; TMUL; TDIVS; TADDS; TMIN
        // TLOAD: Operation
        // TLOAD: Operation
        Sub(diff, pred, target, 64);
        Abs(abs_diff, diff, 64);
        Mul(sq_diff, diff, diff, 64);
        Divs(l2_part, sq_diff, 2.0f, 64);
        Adds(l1_part, abs_diff, -0.5f, 64);
        Min(loss, l2_part, l1_part, 64);

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

extern "C" __global__ __aicore__ void F_smooth_l1_loss_kernel(GM_ADDR input, GM_ADDR output) {
    F_smooth_l1_lossKernel op;
    op.Init(input, output);
    op.Process();
}