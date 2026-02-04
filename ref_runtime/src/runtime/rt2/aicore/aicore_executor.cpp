#include "aicore/aicore.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"
#include <cstdlib>
#include <cstdio>

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Execute task from PTO2DispatchPayload (runtime2 merge path).
 *
 * When merging runtime2 into rt2, Handshake.task points to PTO2DispatchPayload.
 * Unpack: function_bin_addr, args[], num_args; run kernel(args); on completion
 * AICPU uses payload.task_id to push to completion_queue.
 */
__aicore__ __attribute__((always_inline)) static void execute_task_from_payload(__gm__ PTO2DispatchPayload* payload) {
    if (payload == nullptr) {
        return;
    }
    if (payload->function_bin_addr == 0) {
        return;
    }
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(payload->args));
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    (void)core_type;
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready (use core_id + 1 to avoid 0)
    my_hank->aicore_done = block_idx + 1;

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0).
        // Rt2 AICore always receives PTO2DispatchPayload* from AICPU (no host Runtime calls on device).
        if (my_hank->task_status == 1 && my_hank->task != 0) {
            __gm__ PTO2DispatchPayload* payload = reinterpret_cast<__gm__ PTO2DispatchPayload*>(my_hank->task);
            execute_task_from_payload(payload);
            my_hank->task_status = 0;
        }
    }
}
