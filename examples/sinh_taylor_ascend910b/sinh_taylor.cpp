// PTO Program: sinh_taylor
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class sinh_taylorKernel {
public:
    __aicore__ inline sinh_taylorKernel() {}
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

        // Loop fusion: 22 loop overheads saved

        // FUSED (23 ops): TLOAD; TMULS; TMUL; TMULS; TMUL; TDIVS; TADD; TMUL; TDIVS; TADD; TMUL; TDIVS; TADD; TMUL; TDIVS; TADD; TMUL; TDIVS; TADD; TMUL; TDIVS; TADD; TSTORE
        // TLOAD: Operation
        Muls(result, x, 1.0f, 64);
        Mul(x_squared, x, x, 64);
        Muls(term, x, 1.0f, 64);
        Mul(term, term, x_squared, 64);
        Divs(term, term, 6.0f, 64);
        Add(result, result, term, 64);
        Mul(term, term, x_squared, 64);
        Divs(term, term, 20.0f, 64);
        Add(result, result, term, 64);
        Mul(term, term, x_squared, 64);
        Divs(term, term, 42.0f, 64);
        Add(result, result, term, 64);
        Mul(term, term, x_squared, 64);
        Divs(term, term, 72.0f, 64);
        Add(result, result, term, 64);
        Mul(term, term, x_squared, 64);
        Divs(term, term, 110.0f, 64);
        Add(result, result, term, 64);
        Mul(term, term, x_squared, 64);
        Divs(term, term, 156.0f, 64);
        Add(result, result, term, 64);
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

extern "C" __global__ __aicore__ void sinh_taylor_kernel(GM_ADDR input, GM_ADDR output) {
    sinh_taylorKernel op;
    op.Init(input, output);
    op.Process();
}