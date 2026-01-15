// PTO Program: F_elu
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_eluKernel {
public:
    __aicore__ inline F_eluKernel() {}
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

        // Loop fusion: 11 loop overheads saved

        // FUSED (12 ops): TLOAD; TRELU; TEXP; TADDS; TMULS; TNEG; TRELU; TNEG; TMUL; TDIVS; TADD; TSTORE
        // TLOAD: Operation
        Relu(pos_part, x, 64);
        Exp(exp_x, x, 64);
        Adds(exp_minus_one, exp_x, -1.0f, 64);
        Muls(scaled, exp_minus_one, 1.0f, 64);
        Neg(neg_x, x, 64);
        Relu(neg_relu, neg_x, 64);
        Neg(neg_part, neg_relu, 64);
        Mul(neg_scaled, scaled, neg_part, 64);
        Divs(neg_scaled, neg_scaled, 1.0f, 64);
        Add(result, pos_part, neg_scaled, 64);
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

extern "C" __global__ __aicore__ void F_elu_kernel(GM_ADDR input, GM_ADDR output) {
    F_eluKernel op;
    op.Init(input, output);
    op.Process();
}