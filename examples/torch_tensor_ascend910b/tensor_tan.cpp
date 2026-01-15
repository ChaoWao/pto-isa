// PTO Program: tensor_tan
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class tensor_tanKernel {
public:
    __aicore__ inline tensor_tanKernel() {}
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

        // Loop fusion: 15 loop overheads saved

        // FUSED (16 ops): TLOAD; TMUL; TMUL; TMUL; TMUL; TDIVS; TDIVS; TSUB; TADD; TDIVS; TDIVS; TEXPANDS; TSUB; TADD; TDIV; TSTORE
        // TLOAD: Operation
        Mul(x2, x, x, 64);
        Mul(x3, x2, x, 64);
        Mul(x4, x2, x2, 64);
        Mul(x5, x3, x2, 64);
        Divs(sin_t1, x3, 6.0f, 64);
        Divs(sin_t2, x5, 120.0f, 64);
        Sub(sin_temp, x, sin_t1, 64);
        Add(sin_val, sin_temp, sin_t2, 64);
        Divs(cos_t1, x2, 2.0f, 64);
        Divs(cos_t2, x4, 24.0f, 64);
        Duplicate(ones, 1.0f, 64);
        Sub(cos_temp, ones, cos_t1, 64);
        Add(cos_val, cos_temp, cos_t2, 64);
        Div(result, sin_val, cos_val, 64);
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

extern "C" __global__ __aicore__ void tensor_tan_kernel(GM_ADDR input, GM_ADDR output) {
    tensor_tanKernel op;
    op.Init(input, output);
    op.Process();
}