/**
 * PTO Runtime - AICore Kernel
 * 
 * This kernel runs on AICore (both AIC/Cube and AIV/Vector).
 * Compiled with ccec using:
 *   - AIC: --cce-aicore-arch=dav-c220-cube -D__AIC__
 *   - AIV: --cce-aicore-arch=dav-c220-vec -D__AIV__
 * 
 * Communication with AICPU via handshake buffer in global memory.
 * Task execution via function pointer dispatch from functionBinAddr.
 */

#include <cstdint>
#include "pto_task.h"

// =============================================================================
// AICore Attributes and Definitions
// =============================================================================

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// Kernel entry naming based on core type
#ifdef __AIV__
#define KERNEL_ENTRY(x) x##_0_mix_aiv
#define blockIdx blockIdx_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#define blockIdx blockIdx_aic
#endif

[[block_local]] int blockIdx;

// =============================================================================
// Unified Kernel Function Pointer Type
// =============================================================================

/**
 * All InCore kernels follow the unified signature:
 *   void kernel(__gm__ int64_t* args)
 * 
 * This enables switch-free dispatch via function pointer.
 */
typedef void (*PTOUnifiedKernelFunc)(__gm__ int64_t*);

// =============================================================================
// Task Execution
// =============================================================================

/**
 * Execute a single task using function pointer dispatch.
 * 
 * The functionBinAddr in the task points to the compiled kernel
 * binary in device GM memory. We cast it to a function pointer
 * and invoke with the task's argument array.
 * 
 * @param task Pointer to task in global memory
 */
__aicore__ __attribute__((always_inline)) 
static void execute_task(__gm__ PTOTask* task) {
    if (task == nullptr) {
        return;
    }
    
    // Check for valid function binary address
    if (task->functionBinAddr == 0) {
        return;
    }
    
    // Cast functionBinAddr to unified function pointer and invoke
    PTOUnifiedKernelFunc kernel = (PTOUnifiedKernelFunc)task->functionBinAddr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

// =============================================================================
// AICore Kernel Entry Point
// =============================================================================

/**
 * AICore kernel entry point with control loop.
 * 
 * Protocol:
 * 1. Wait for AICPU ready signal (handshake initialization)
 * 2. Signal AICore is ready (aicore_done = core_id + 1)
 * 3. Enter polling loop:
 *    - Check control flag (1 = quit, 0 = continue)
 *    - If task pointer is non-zero, execute task and mark complete
 *    - Use DCCI to ensure cache coherency with AICPU
 * 
 * Each core gets its own handshake buffer indexed by blockIdx.
 * 
 * @param hank Array of handshake buffers (one per core)
 */
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ PTOHandshake* hank) {
    // Calculate blockIdx for this core
#ifdef __AIV__
    // AIV cores are indexed after AIC cores
    blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    // AIC cores are indexed from 0
    blockIdx = get_block_idx();
#endif

    // Get this core's handshake buffer
    __gm__ PTOHandshake* my_hank = &hank[blockIdx];

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready (use core_id + 1 to avoid 0)
    my_hank->aicore_done = blockIdx + 1;

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        // Invalidate cache to get fresh data from AICPU
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;
        }

        // Execute task if assigned (task != 0 means valid PTOTask* pointer)
        if (my_hank->task != 0) {
            __gm__ PTOTask* task_ptr = reinterpret_cast<__gm__ PTOTask*>(my_hank->task);
            execute_task(task_ptr);
            
            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
    }
}
