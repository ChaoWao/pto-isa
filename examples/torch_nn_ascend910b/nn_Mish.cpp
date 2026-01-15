// PTO Program: nn_Mish
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class nn_MishKernel {
public:
    __aicore__ inline nn_MishKernel() {}
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

        // Loop fusion: 11 loop overheads saved

        // FUSED (12 ops): TLOAD; TEXP; TADDS; TLOG; TEXP; TNEG; TEXP; TSUB; TADD; TDIV; TMUL; TSTORE
        // TLOAD: Operation
        Exp(exp_x, x, 64);
        Adds(one_plus_exp, exp_x, 1.0f, 64);
        Ln(softplus, one_plus_exp, 64);
        Exp(exp_sp, softplus, 64);
        Neg(neg_sp, softplus, 64);
        Exp(exp_neg_sp, neg_sp, 64);
        Sub(tanh_num, exp_sp, exp_neg_sp, 64);
        Add(tanh_den, exp_sp, exp_neg_sp, 64);
        Div(tanh_out, tanh_num, tanh_den, 64);
        Mul(result, x, tanh_out, 64);
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

extern "C" __global__ __aicore__ void nn_Mish_kernel(GM_ADDR input, GM_ADDR output) {
    nn_MishKernel op;
    op.Init(input, output);
    op.Process();
}