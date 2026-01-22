/**
 * Minimal AICore Kernel with PTO Support
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __out__
#define __out__
#endif

#ifdef __AIV__
#define KERNEL_ENTRY(x) x##_0_mix_aiv   // 动态生成函数名 KERNEL_ENTRY(my_kernel) -> my_kernel_0_mix_aiv
#define blockIdx blockIdx_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#define blockIdx blockIdx_aic
#endif

[[block_local]] int blockIdx;

using namespace pto;

struct Handshake {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile int32_t control;  // 0=execute, 1=quit
    volatile int32_t task;     // task ID: -1=none, 0=TADD, etc.
} __attribute__((aligned(64)));

// TADD implementation (float path)
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__aicore__ __attribute__((always_inline)) void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

// Task execution wrapper
__aicore__ __attribute__((always_inline)) static void execute_task(int32_t task_id)
{
    switch(task_id) {
        case 0:  // TADD task
#ifdef __AIV__
                // runTAdd<float, 64, 64, 64, 64>(out, src0, src1);
#endif
            break;
        default:
            // Unknown task, do nothing
            break;
    }
}

/**
 * Kernel entry point with control loop
 *
 * This function is called by the runtime when kernel is launched.
 * It waits for tasks from AICPU and executes them based on control flags.
 * Each core (AIC or AIV) gets its own handshake buffer indexed by blockIdx.
 */
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ struct Handshake* hank) {
    // Calculate blockIdx for this core
#ifdef __AIV__
    blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    blockIdx = get_block_idx();
#endif

    // Get this core's handshake buffer
    __gm__ Handshake* my_hank = &hank[blockIdx];

    // Wait for AICPU signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }
    // Signal completion
    my_hank->aicore_done = blockIdx + 1;
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        // Read control flag
        if (my_hank->control == 1) {
            break;  // Quit command
        }

        // Execute task if valid
        if (my_hank->task != -1) {
            execute_task(my_hank->task);
        }
    }
}
