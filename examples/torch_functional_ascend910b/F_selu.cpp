// PTO Program: F_selu
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_seluKernel {
public:
    __aicore__ inline F_seluKernel() {}
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

        // Loop fusion: 9 loop overheads saved

        // FUSED (10 ops): TLOAD; TRELU; TEXP; TADDS; TMULS; TEXPANDS; TMIN; TADD; TMULS; TSTORE
        // TLOAD: Operation
        Relu(pos_part, x, 64);
        Exp(exp_x, x, 64);
        Adds(exp_minus_one, exp_x, -1.0f, 64);
        Muls(alpha_scaled, exp_minus_one, 1.6732632423543772f, 64);
        Duplicate(zeros, 0.0f, 64);
        Min(neg_part, alpha_scaled, zeros, 64);
        Add(elu_result, pos_part, neg_part, 64);
        Muls(result, elu_result, 1.0507009873554805f, 64);
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

extern "C" __global__ __aicore__ void F_selu_kernel(GM_ADDR input, GM_ADDR output) {
    F_seluKernel op;
    op.Init(input, output);
    op.Process();
}