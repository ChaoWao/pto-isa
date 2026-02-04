/**
 * Example: host_orchestration_entry - Host-side orchestration for rt2
 *
 * Same DAG as aicpu_orchestration_entry; runs on host and writes into
 * host_sm_mirror. Caller then copies host_mirror to device SM.
 *   t0: c = a + b, t1: d = c + 1, t2: e = c + 2, t3: f = d * e
 *
 * Compiled into host runtime only. args layout: [dev_a, dev_b, dev_f,
 * size_a, size_b, size_f, SIZE, dev_c, dev_d, dev_e [, gm_heap, heap_size]]
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <cstdlib>

#include "pto_runtime2.h"
#include "pto_shared_memory.h"

static void debug_tensor_first16(const char* label, const void* ptr, size_t size) {
    if (!ptr || std::getenv("PTO2_DEBUG_TENSOR") == nullptr) return;
    /* ptr is a device address; host must not dereference it (would segfault). */
    (void)size;
    fprintf(stderr, "[Orchestrator/Host] %s=%p (device ptr, first16 not read on host)\n", label, ptr);
}

#define ARG_DEV_A  0
#define ARG_DEV_B  1
#define ARG_DEV_F  2
#define ARG_SIZE_A 3
#define ARG_SIZE_B 4
#define ARG_SIZE_F 5
#define ARG_SIZE   6
#define ARG_DEV_C  7
#define ARG_DEV_D  8
#define ARG_DEV_E  9

#ifndef PTO2_TASK_WINDOW_SIZE
#define PTO2_TASK_WINDOW_SIZE 16384
#endif
#ifndef PTO2_DEP_LIST_POOL_SIZE
#define PTO2_DEP_LIST_POOL_SIZE 65536
#endif
#ifndef PTO2_HEAP_SIZE
#define PTO2_HEAP_SIZE (256 * 1024)
#endif

/** Called from host after allocating host_sm_mirror (same layout as PTO2 SM). */
extern "C" void host_orchestration_entry(void* host_sm_mirror, uint64_t* args, int arg_count) {
    if (!host_sm_mirror || !args || arg_count < 7) {
        if (host_sm_mirror) {
            *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;  /* orchestrator_done */
        }
        return;
    }

    void* dev_a = (void*)(uintptr_t)args[ARG_DEV_A];
    void* dev_b = (void*)(uintptr_t)args[ARG_DEV_B];
    void* dev_f = (void*)(uintptr_t)args[ARG_DEV_F];
    size_t size_a = (size_t)args[ARG_SIZE_A];
    size_t size_b = (size_t)args[ARG_SIZE_B];
    size_t size_f = (size_t)args[ARG_SIZE_F];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    void* dev_c = arg_count >= 10 ? (void*)(uintptr_t)args[ARG_DEV_C] : nullptr;
    void* dev_d = arg_count >= 10 ? (void*)(uintptr_t)args[ARG_DEV_D] : nullptr;
    void* dev_e = arg_count >= 10 ? (void*)(uintptr_t)args[ARG_DEV_E] : nullptr;

    if (!dev_c || !dev_d || !dev_e) {
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    size_t BYTES = (size_t)SIZE * sizeof(float);
    int32_t sm_size = pto2_sm_calculate_size(PTO2_TASK_WINDOW_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    PTO2SharedMemoryHandle* sm_handle = pto2_sm_create_from_buffer(
        host_sm_mirror,
        sm_size,
        PTO2_TASK_WINDOW_SIZE,
        PTO2_HEAP_SIZE,
        PTO2_DEP_LIST_POOL_SIZE
    );
    if (!sm_handle) {
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    void* gm_heap = nullptr;
    int32_t heap_size = 0;
    if (arg_count >= 12 && args[10] != 0 && args[11] != 0) {
        gm_heap = (void*)(uintptr_t)args[10];
        heap_size = (int32_t)(args[11] & 0x7FFFFFFF);
    }
    if (!gm_heap || heap_size <= 0) {
        pto2_sm_destroy(sm_handle);
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    PTO2Runtime* rt = pto2_runtime_create_from_sm(
        PTO2_MODE_EXECUTE,
        sm_handle,
        gm_heap,
        heap_size
    );
    if (!rt) {
        pto2_sm_destroy(sm_handle);
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    int32_t tile = 0;
    int32_t sz = (int32_t)BYTES;
    if (sz <= 0) sz = (int32_t)size_a;

    debug_tensor_first16("dev_a", dev_a, size_a);
    debug_tensor_first16("dev_b", dev_b, size_b);

    PTO2_SCOPE_BEGIN(rt);

    enum { PTO2_PARAM_STRIDE = 24 };
    static alignas(8) unsigned char params_t0_buf[3 * PTO2_PARAM_STRIDE];
    pto2_param_set_input((PTO2TaskParam*)(params_t0_buf + 0),  dev_a, tile, sz);
    pto2_param_set_input((PTO2TaskParam*)(params_t0_buf + 24), dev_b, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t0_buf + 48), dev_c, tile, sz);
    pto2_param_fix_sizes(params_t0_buf, 3, sz);
    if (pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, (PTO2TaskParam*)params_t0_buf, 3) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    static alignas(8) unsigned char params_t1_buf[2 * PTO2_PARAM_STRIDE];
    pto2_param_set_input((PTO2TaskParam*)(params_t1_buf + 0),  dev_c, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t1_buf + 24), dev_d, tile, sz);
    pto2_param_fix_sizes(params_t1_buf, 2, sz);
    if (pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, (PTO2TaskParam*)params_t1_buf, 2) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    static alignas(8) unsigned char params_t2_buf[2 * PTO2_PARAM_STRIDE];
    pto2_param_set_input((PTO2TaskParam*)(params_t2_buf + 0),  dev_c, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t2_buf + 24), dev_e, tile, sz);
    pto2_param_fix_sizes(params_t2_buf, 2, sz);
    if (pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, (PTO2TaskParam*)params_t2_buf, 2) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    static alignas(8) unsigned char params_t3_buf[3 * PTO2_PARAM_STRIDE];
    pto2_param_set_input((PTO2TaskParam*)(params_t3_buf + 0),  dev_d, tile, sz);
    pto2_param_set_input((PTO2TaskParam*)(params_t3_buf + 24), dev_e, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t3_buf + 48), dev_f, tile, sz);
    pto2_param_fix_sizes(params_t3_buf, 3, sz);
    int32_t task3_id = pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, (PTO2TaskParam*)params_t3_buf, 3);
    if (task3_id < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);
        *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
        return;
    }

    void* graph_out_ptr = pto2_rt_get_output(rt, task3_id, 0);
    debug_tensor_first16("output(graph_out_ptr)", graph_out_ptr, size_f);
    if (graph_out_ptr && size_f > 0) {
        rt->sm_handle->header->graph_output_ptr = (uint64_t)(uintptr_t)graph_out_ptr;
        rt->sm_handle->header->graph_output_size = (int32_t)size_f;
    }

    PTO2_SCOPE_END(rt);
    pto2_rt_orchestration_done(rt);
    pto2_runtime_destroy(rt);
    *(volatile int32_t*)((char*)host_sm_mirror + 8) = 1;
}
