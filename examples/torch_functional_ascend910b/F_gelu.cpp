// PTO Program: F_gelu
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_geluKernel {
public:
    __aicore__ inline F_geluKernel() {}
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

        // Loop fusion: 14 loop overheads saved

        // FUSED (15 ops): TLOAD; TMUL; TMUL; TMULS; TADD; TMULS; TMULS; TEXP; TADDS; TADDS; TDIV; TADDS; TMULS; TMUL; TSTORE
        // TLOAD: Operation
        Mul(x_sq, x, x, 64);
        Mul(x_cubed, x_sq, x, 64);
        Muls(coeff_x3, x_cubed, 0.044715f, 64);
        Add(inner, x, coeff_x3, 64);
        Muls(scaled, inner, 0.7978845608028654f, 64);
        Muls(scaled, scaled, 2.0f, 64);
        Exp(exp_pos, scaled, 64);
        Adds(sinh_approx, exp_pos, -1.0f, 64);
        Adds(cosh_approx, exp_pos, 1.0f, 64);
        Div(tanh_out, sinh_approx, cosh_approx, 64);
        Adds(one_plus, tanh_out, 1.0f, 64);
        Muls(half_x, x, 0.5f, 64);
        Mul(result, half_x, one_plus, 64);
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

extern "C" __global__ __aicore__ void F_gelu_kernel(GM_ADDR input, GM_ADDR output) {
    F_geluKernel op;
    op.Init(input, output);
    op.Process();
}