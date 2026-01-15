// PTO Program: tensor_std
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class tensor_stdKernel {
public:
    __aicore__ inline tensor_stdKernel() {}
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

        // Loop fusion: 4 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // BARRIER: TROWSUM

        // BARRIER: TCOLSUM

        // FUSED (1 ops): TDIVS
        Divs(total, total, 64.0f, 64);

        // FUSED (3 ops): TEXPANDS; TSUB; TMUL
        Duplicate(mean_val, 0.0f, 64);
        Sub(centered, self, mean_val, 64);
        Mul(sq_centered, centered, centered, 64);

        // BARRIER: TROWSUM

        // BARRIER: TCOLSUM

        // FUSED (3 ops): TDIVS; TSQRT; TSTORE
        Divs(var, var_total, 64.0f, 64);
        Sqrt(result, var, 64);
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

extern "C" __global__ __aicore__ void tensor_std_kernel(GM_ADDR input, GM_ADDR output) {
    tensor_stdKernel op;
    op.Init(input, output);
    op.Process();
}