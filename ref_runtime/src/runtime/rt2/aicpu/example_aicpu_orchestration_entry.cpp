/**
 * Example: aicpu_orchestration_entry 设备端编排
 *
 * 与 example_orch.cpp 中 build_example_graph 对齐同一 DAG：
 *   t0: c = a + b     (func_id=0, kernel_add)
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar)
 *   t3: f = d * e     (func_id=2, kernel_mul)
 *   依赖: t0→t1, t0→t2, t1→t3, t2→t3
 *
 * 编译：AICPU 构建包含 runtime2，本文件与 device_orchestration_stub (weak) 同编；
 * 本文件提供强符号 aicpu_orchestration_entry，链接时覆盖 stub。
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <cstdlib>

#include "pto_runtime2.h"

static void debug_tensor_first16(const char* label, const void* ptr, size_t size) {
    if (!ptr || std::getenv("PTO2_DEBUG_TENSOR") == nullptr) return;
    const unsigned char* p = static_cast<const unsigned char*>(ptr);
    size_t n = size < 16 ? size : 16;
    fprintf(stderr, "[Orchestrator] %s=%p first16=", label, ptr);
    for (size_t i = 0; i < n; i++) fprintf(stderr, "%02x", p[i]);
    fprintf(stderr, "\n");
}
#include "pto_shared_memory.h"

// args 布局：[dev_a, dev_b, dev_f, size_a, size_b, size_f, SIZE, dev_c, dev_d, dev_e]
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
/* Need space for 4 task outputs: each SIZE*sizeof(float); SIZE=16384 -> 64KB each -> 256KB total */
#ifndef PTO2_HEAP_SIZE
#define PTO2_HEAP_SIZE (256 * 1024)
#endif

static char s_gm_heap_stub[PTO2_HEAP_SIZE];

extern "C" {

void aicpu_orchestration_entry(void* sm_ptr, uint64_t* args, int arg_count) {
    if (!sm_ptr || !args || arg_count < 7) {
        if (sm_ptr) {
            *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
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
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    size_t BYTES = (size_t)SIZE * sizeof(float);

    int32_t sm_size = pto2_sm_calculate_size(PTO2_TASK_WINDOW_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    PTO2SharedMemoryHandle* sm_handle = pto2_sm_create_from_buffer(
        sm_ptr,
        sm_size,
        PTO2_TASK_WINDOW_SIZE,
        PTO2_HEAP_SIZE,
        PTO2_DEP_LIST_POOL_SIZE
    );
    if (!sm_handle) {
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    void* gm_heap = s_gm_heap_stub;
    int32_t heap_size = (int32_t)sizeof(s_gm_heap_stub);
    if (arg_count >= 12 && args[10] != 0 && args[11] != 0) {
        gm_heap = (void*)(uintptr_t)args[10];
        heap_size = (int32_t)(args[11] & 0x7FFFFFFF);
    }

    PTO2Runtime* rt = pto2_runtime_create_from_sm(
        PTO2_MODE_EXECUTE,
        sm_handle,
        gm_heap,
        heap_size
    );
    if (!rt) {
        /* sm_handle not yet passed to rt; we own it */
        pto2_sm_destroy(sm_handle);
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    int32_t tile = 0;
    int32_t sz = (int32_t)BYTES;
    if (sz <= 0) sz = (int32_t)size_a;

    debug_tensor_first16("dev_a", dev_a, size_a);
    debug_tensor_first16("dev_b", dev_b, size_b);

    PTO2_SCOPE_BEGIN(rt);

    /* Use C helpers so param layout (size at offset 20) matches orchestrator */
    enum { PTO2_PARAM_STRIDE = 24 };
    static unsigned char params_t0_buf[3 * PTO2_PARAM_STRIDE] __attribute__((aligned(8)));  /* static to avoid stack reuse */
    pto2_param_set_input((PTO2TaskParam*)(params_t0_buf + 0),  dev_a, tile, sz);
    pto2_param_set_input((PTO2TaskParam*)(params_t0_buf + 24), dev_b, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t0_buf + 48), dev_c, tile, sz);
    pto2_param_fix_sizes(params_t0_buf, 3, sz);  /* C-side write so C-side read sees correct size */
    if (pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, nullptr, "kernel_add", (PTO2TaskParam*)params_t0_buf, 3) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  /* destroys sm_handle internally */
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    static unsigned char params_t1_buf[2 * PTO2_PARAM_STRIDE] __attribute__((aligned(8)));
    pto2_param_set_input((PTO2TaskParam*)(params_t1_buf + 0),  dev_c, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t1_buf + 24), dev_d, tile, sz);
    pto2_param_fix_sizes(params_t1_buf, 2, sz);
    if (pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, nullptr, "kernel_add_scalar", (PTO2TaskParam*)params_t1_buf, 2) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  /* destroys sm_handle internally */
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    static unsigned char params_t2_buf[2 * PTO2_PARAM_STRIDE] __attribute__((aligned(8)));
    pto2_param_set_input((PTO2TaskParam*)(params_t2_buf + 0),  dev_c, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t2_buf + 24), dev_e, tile, sz);
    pto2_param_fix_sizes(params_t2_buf, 2, sz);
    if (pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, nullptr, "kernel_add_scalar", (PTO2TaskParam*)params_t2_buf, 2) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  /* destroys sm_handle internally */
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    static unsigned char params_t3_buf[3 * PTO2_PARAM_STRIDE] __attribute__((aligned(8)));
    pto2_param_set_input((PTO2TaskParam*)(params_t3_buf + 0),  dev_d, tile, sz);
    pto2_param_set_input((PTO2TaskParam*)(params_t3_buf + 24), dev_e, tile, sz);
    pto2_param_set_output((PTO2TaskParam*)(params_t3_buf + 48), dev_f, tile, sz);
    pto2_param_fix_sizes(params_t3_buf, 3, sz);
    int32_t task3_id = pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, nullptr, "kernel_mul", (PTO2TaskParam*)params_t3_buf, 3);
    if (task3_id < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  /* destroys sm_handle internally */
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }
    /* Set graph output for copy-back: host will copy from packed buffer, not dev_f */
    void* graph_out_ptr = pto2_rt_get_output(rt, task3_id, 0);
    debug_tensor_first16("output(graph_out_ptr)", graph_out_ptr, size_f);
    if (graph_out_ptr && size_f > 0) {
        rt->sm_handle->header->graph_output_ptr = (uint64_t)(uintptr_t)graph_out_ptr;
        rt->sm_handle->header->graph_output_size = (int32_t)size_f;
    }

    PTO2_SCOPE_END(rt);
    pto2_rt_orchestration_done(rt);

    pto2_runtime_destroy(rt);  /* destroys sm_handle internally; do not call pto2_sm_destroy(sm_handle) */

    *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
}

}  // extern "C"
