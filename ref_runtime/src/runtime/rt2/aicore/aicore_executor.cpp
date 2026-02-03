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
 * Task execution wrapper - dispatches tasks using function pointers
 *
 * This function demonstrates the runtime function pointer dispatch pattern.
 * Following the production system flow:
 * - function_bin_addr points to compiled kernel code in device GM memory
 * - The address is cast to a function pointer: UnifiedKernelFunc kernel =
 * (UnifiedKernelFunc)function_bin_addr
 * - The kernel is invoked: kernel(task->args)
 *
 * This is the KEY difference from compile-time linking:
 * - OLD: extern "C" declarations, resolved at link time
 * - NEW: function_bin_addr from GM memory, cast at runtime
 *
 * With unified kernel signature, no switch statement is needed.
 * All kernels unpack their own arguments from the args array.
 *
 * @param task Pointer to task in global memory (null during initialization)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task) {
    // Null task pointer indicates no work assigned (initialization state)
    if (task == nullptr) {
        return;
    }

    // Check for valid function_bin_addr
    if (task->function_bin_addr == 0) {
        // Invalid address - skip execution
        return;
    }

    // Cast function_bin_addr to unified function pointer and invoke
    // All kernels have signature: void kernel(__gm__ int64_t* args)
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

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
#if !defined(__PLATFORM_A2A3__) || defined(__CPU_SIM)
    if (getenv("PTO2_DEBUG_TENSOR") != nullptr && payload->num_args >= 3) {
        uintptr_t a0 = (uintptr_t)payload->args[0];
        uintptr_t a1 = (uintptr_t)payload->args[1];
        uintptr_t a2 = (uintptr_t)payload->args[2];
        /* Only dereference as pointer if value looks like a real pointer (e.g. has high bits); 32-bit scalar (e.g. float 1.0f = 0x3f800000) must not be dereferenced */
        int ptr_like_0 = (a0 >= 0x1000 && (a0 >> 32) != 0);
        int ptr_like_1 = (a1 >= 0x1000 && (a1 >> 32) != 0);
        int ptr_like_2 = (a2 >= 0x1000 && (a2 >> 32) != 0);
        fprintf(stderr, "[Worker/AICore] task_id=%d input0=%p first16=", payload->task_id, (void*)a0);
        if (ptr_like_0) {
            const unsigned char* p0 = (const unsigned char*)a0;
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", p0[i]);
        } else fprintf(stderr, "(skip)");
        fprintf(stderr, " input1=%p first16=", (void*)a1);
        if (ptr_like_1) {
            const unsigned char* p1 = (const unsigned char*)a1;
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", p1[i]);
        } else fprintf(stderr, "(scalar/skip)");
        fprintf(stderr, " output=%p first16=", (void*)a2);
        if (ptr_like_2) {
            const unsigned char* p2 = (const unsigned char*)a2;
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", p2[i]);
        } else fprintf(stderr, "(skip)");
        fprintf(stderr, "\n");
    }
#endif
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(payload->args));
#if !defined(__PLATFORM_A2A3__) || defined(__CPU_SIM)
    if (getenv("PTO2_DEBUG_TENSOR") != nullptr && payload->num_args >= 3) {
        uintptr_t a2 = (uintptr_t)payload->args[2];
        fprintf(stderr, "[Worker/AICore] task_id=%d output after kernel first16=", payload->task_id);
        if (a2 >= 0x1000) {
            const unsigned char* p2 = (const unsigned char*)a2;
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", p2[i]);
        } else fprintf(stderr, "(skip)");
        fprintf(stderr, "\n");
    }
#endif
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
        // Device PTO2 mode: Handshake.task = PTO2DispatchPayload*; use execute_task_from_payload.
        // Host mode: Handshake.task = Task*; use execute_task.
        if (my_hank->task_status == 1 && my_hank->task != 0) {
            if (runtime->use_pto2_dispatch_) {
                __gm__ PTO2DispatchPayload* payload = reinterpret_cast<__gm__ PTO2DispatchPayload*>(my_hank->task);
                execute_task_from_payload(payload);
            } else {
                __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);
                execute_task(task_ptr);
            }
            my_hank->task_status = 0;
        }
    }
}
