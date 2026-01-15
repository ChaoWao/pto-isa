// PTO Program: tensor_acos
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class tensor_acosKernel {
public:
    __aicore__ inline tensor_acosKernel() {}
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

        // Loop fusion: 10 loop overheads saved

        // FUSED (11 ops): TLOAD; TMUL; TMUL; TMUL; TDIVS; TMULS; TADD; TADD; TEXPANDS; TSUB; TSTORE
        // TLOAD: Operation
        Mul(x2, x, x, 64);
        Mul(x3, x2, x, 64);
        Mul(x5, x3, x2, 64);
        Divs(term1, x3, 6.0f, 64);
        Muls(term2, x5, 0.075f, 64);
        Add(temp, x, term1, 64);
        Add(asin_val, temp, term2, 64);
        Duplicate(pi_half, 1.5707963267948966f, 64);
        Sub(result, pi_half, asin_val, 64);
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

extern "C" __global__ __aicore__ void tensor_acos_kernel(GM_ADDR input, GM_ADDR output) {
    tensor_acosKernel op;
    op.Init(input, output);
    op.Process();
}