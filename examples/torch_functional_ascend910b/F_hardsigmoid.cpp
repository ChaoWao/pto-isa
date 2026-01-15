// PTO Program: F_hardsigmoid
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_hardsigmoidKernel {
public:
    __aicore__ inline F_hardsigmoidKernel() {}
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

        // Loop fusion: 7 loop overheads saved

        // FUSED (8 ops): TLOAD; TADDS; TDIVS; TEXPANDS; TEXPANDS; TMAX; TMIN; TSTORE
        // TLOAD: Operation
        Adds(x_plus_3, x, 3.0f, 64);
        Divs(scaled, x_plus_3, 6.0f, 64);
        Duplicate(zeros, 0.0f, 64);
        Duplicate(ones, 1.0f, 64);
        Max(clamp_low, scaled, zeros, 64);
        Min(result, clamp_low, ones, 64);
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

extern "C" __global__ __aicore__ void F_hardsigmoid_kernel(GM_ADDR input, GM_ADDR output) {
    F_hardsigmoidKernel op;
    op.Init(input, output);
    op.Process();
}