/**
 * Example: aicpu_orchestration_entry orchestration entry
 *
 * DAG structure for formula: (a + b + 1)(a + b + 2)
 *   t0: c = a + b     (func_id=0, kernel_add)
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar)
 *   t3: f = d * e     (func_id=2, kernel_mul)
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3
 *
 * Compilation notes:
 *   - Add this file to AICPU build and link with rt2 runtime .c/.h
 *   - Or copy to ref_runtime/src/runtime/rt2/aicpu/ and configure CUSTOM_SOURCE_DIRS to include runtime
 * Include path example: -I$(REF_RUNTIME)/src/runtime/rt2/runtime
 */

#include <stdint.h>
#include <stddef.h>

// Runtime2 头文件（按实际工程调整 include 路径）
#include "pto_runtime2.h"
#include "pto_shared_memory.h"

// =============================================================================
// args 布局（与 example_orch.cpp / golden.py 一致）
// =============================================================================
// args[0]  = dev_a     (GM 指针)
// args[1]  = dev_b     (GM 指针)
// args[2]  = dev_f     (GM 指针，输出)
// args[3]  = size_a    (bytes)
// args[4]  = size_b    (bytes)
// args[5]  = size_f    (bytes)
// args[6]  = SIZE      (元素个数)
// 设备端扩展（由 host 在 use_device_orchestration 时传入）：
// args[7]  = dev_c     (GM 指针，中间张量)
// args[8]  = dev_d     (GM 指针，中间张量)
// args[9]  = dev_e     (GM 指针，中间张量)
// 若 arg_count >= 12：args[10]=gm_heap, args[11]=heap_size（可选）
// =============================================================================

#ifndef PTO2_TASK_WINDOW_SIZE
#define PTO2_TASK_WINDOW_SIZE 16384
#endif
#ifndef PTO2_DEP_LIST_POOL_SIZE
#define PTO2_DEP_LIST_POOL_SIZE 65536
#endif
#ifndef PTO2_HEAP_SIZE
#define PTO2_HEAP_SIZE (64 * 1024)
#endif

/* 设备端无堆时使用的静态缓冲区（仅 sim/单进程可用；真机需 host 传 gm_heap） */
static char s_gm_heap_stub[PTO2_HEAP_SIZE];

extern "C" {

__attribute__((visibility("default")))
void aicpu_orchestration_entry(void* sm_ptr, uint64_t* args, int arg_count) {
    if (!sm_ptr || !args || arg_count < 7) {
        /* 无效参数时仅打完成标记，与 stub 行为一致 */
        if (sm_ptr) {
            *(volatile int32_t*)((char*)sm_ptr + 8) = 1;  /* orchestrator_done */
        }
        return;
    }

    void* dev_a = (void*)(uintptr_t)args[0];
    void* dev_b = (void*)(uintptr_t)args[1];
    void* dev_f = (void*)(uintptr_t)args[2];
    size_t size_a = (size_t)args[3];
    size_t size_b = (size_t)args[4];
    size_t size_f = (size_t)args[5];
    int SIZE = (int)(args[6] & 0x7FFFFFFF);

    /* 中间张量 c,d,e：若 host 传入则用，否则用占位（真机需 host 分配并传入） */
    void* dev_c = arg_count >= 10 ? (void*)(uintptr_t)args[7] : nullptr;
    void* dev_d = arg_count >= 10 ? (void*)(uintptr_t)args[8] : nullptr;
    void* dev_e = arg_count >= 10 ? (void*)(uintptr_t)args[9] : nullptr;

    if (!dev_c || !dev_d || !dev_e) {
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    size_t BYTES = (size_t)SIZE * sizeof(float);

    /* 1) 用已有 GM 缓冲区包装为 shared memory handle */
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

    /* 2) GM 堆：host 可经 args[10],args[11] 传入；否则用静态 stub（仅 sim） */
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
        pto2_sm_destroy(sm_handle);
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    /* 3) 与 example_orch.cpp 完全对应的 4 个 task + 依赖（通过 INPUT/OUTPUT 由 runtime2 推断） */
    int32_t tile = 0;
    int32_t sz = (int32_t)BYTES;
    if (sz <= 0) sz = (int32_t)size_a;

    PTO2_SCOPE_BEGIN(rt);

    /* t0: c = a + b   (kernel_id=0, kernel_add, AIV) */
    PTO2TaskParam params_t0[] = {
        PTO2_INPUT(dev_a, tile, sz),
        PTO2_INPUT(dev_b, tile, sz),
        PTO2_OUTPUT(dev_c, tile, sz),
    };
    if (pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, params_t0, 3) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  // Also destroys sm_handle
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    /* t1: d = c + 1   (kernel_id=1, kernel_add_scalar) — depends on t0's dev_c */
    PTO2TaskParam params_t1[] = {
        PTO2_INPUT(dev_c, tile, sz),
        PTO2_OUTPUT(dev_d, tile, sz),
    };
    if (pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, params_t1, 2) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  // Also destroys sm_handle
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    /* t2: e = c + 2   (kernel_id=1, kernel_add_scalar) — depends on t0's dev_c */
    PTO2TaskParam params_t2[] = {
        PTO2_INPUT(dev_c, tile, sz),
        PTO2_OUTPUT(dev_e, tile, sz),
    };
    if (pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, params_t2, 2) < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  // Also destroys sm_handle
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    /* t3: f = d * e   (kernel_id=2, kernel_mul) — depends on t1's dev_d, t2's dev_e */
    PTO2TaskParam params_t3[] = {
        PTO2_INPUT(dev_d, tile, sz),
        PTO2_INPUT(dev_e, tile, sz),
        PTO2_OUTPUT(dev_f, tile, sz),
    };
    int32_t task3_id = pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, params_t3, 3);
    if (task3_id < 0) {
        PTO2_SCOPE_END(rt);
        pto2_rt_orchestration_done(rt);
        pto2_runtime_destroy(rt);  // Also destroys sm_handle
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    /* Set graph_output_ptr so host can copy from packed buffer */
    void* graph_out_ptr = pto2_rt_get_output(rt, task3_id, 0);
    if (graph_out_ptr && size_f > 0) {
        rt->sm_handle->header->graph_output_ptr = (uint64_t)(uintptr_t)graph_out_ptr;
        rt->sm_handle->header->graph_output_size = (int32_t)size_f;
    }

    PTO2_SCOPE_END(rt);
    pto2_rt_orchestration_done(rt);

    /* 4) 清理（不释放 sm_ptr / gm_heap，由调用方管理）
     * Note: pto2_runtime_destroy already calls pto2_sm_destroy(rt->sm_handle),
     * so we should NOT call pto2_sm_destroy(sm_handle) again to avoid double-free */
    pto2_runtime_destroy(rt);

    /* 打完成标记，供调度线程轮询 */
    *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
}

}  // extern "C"
