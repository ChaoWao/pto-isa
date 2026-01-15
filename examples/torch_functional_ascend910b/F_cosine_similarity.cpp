// PTO Program: F_cosine_similarity
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_cosine_similarityKernel {
public:
    __aicore__ inline F_cosine_similarityKernel() {}
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

        // FUSED (3 ops): TLOAD; TLOAD; TMUL
        // TLOAD: Operation
        // TLOAD: Operation
        Mul(dot_prod, x1, x2, 64);

        // BARRIER: TROWSUM

        // FUSED (2 ops): TMUL; TMUL
        Mul(x1_sq, x1, x1, 64);
        Mul(x2_sq, x2, x2, 64);

        // BARRIER: TROWSUM

        // BARRIER: TROWSUM

        // FUSED (6 ops): TSQRT; TSQRT; TMUL; TADDS; TDIV; TSTORE
        Sqrt(x1_norm, x1_norm_sq, 64);
        Sqrt(x2_norm, x2_norm_sq, 64);
        Mul(norm_prod, x1_norm, x2_norm, 64);
        Adds(norm_prod, norm_prod, 1e-08f, 64);
        Div(result, dot_sum, norm_prod, 64);
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

extern "C" __global__ __aicore__ void F_cosine_similarity_kernel(GM_ADDR input, GM_ADDR output) {
    F_cosine_similarityKernel op;
    op.Init(input, output);
    op.Process();
}