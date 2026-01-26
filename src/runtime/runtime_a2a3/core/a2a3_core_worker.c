/**
 * PTO Runtime - Ascend A2/A3 Hardware Core Worker Implementation
 * 
 * This file implements the worker functions for real Ascend A2/A3 hardware.
 * Requires CANN SDK for actual NPU kernel execution.
 */

#include "a2a3_core_worker.h"
#include "../orchestration/a2a3_orchestration.h"
#include <stdlib.h>
#include <stdio.h>

// =============================================================================
// CANN SDK Requirement Check
// =============================================================================

#if defined(A2A3_TARGET_HARDWARE) && !defined(CANN_SDK_AVAILABLE) && !defined(A2A3_SKIP_CANN_CHECK)
#error "=================================================================="
#error "Ascend A2/A3 Hardware Core Worker requires CANN SDK."
#error ""
#error "To compile for real hardware, you need to:"
#error "  1. Install Huawei CANN SDK (version 6.0 or later)"
#error "  2. Set environment: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
#error "  3. Define CANN_SDK_AVAILABLE when compiling"
#error ""
#error "For simulation/testing without hardware, use:"
#error "  - Platform: ascend_a2a3_sim"
#error "  - Or define A2A3_SKIP_CANN_CHECK for stub-only compilation"
#error "=================================================================="
#endif

// =============================================================================
// Task Execution (Hardware Implementation)
// =============================================================================

#ifdef CANN_SDK_AVAILABLE
#include <acl/acl.h>

void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Core HW] Worker %d executing task %d: %s\n", 
                worker_id, task_id, task->func_name);
    
    // Build argument array
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Execute on NPU via CANN kernel launch
    if (task->func_ptr) {
        // TODO: Replace with actual CANN kernel launch
        // aclrtLaunchKernel(...);
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    }
    
    // Synchronize with NPU
    aclrtSynchronizeStream(NULL);
}

#else /* No CANN SDK - stub implementation for compilation testing */

void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Core HW STUB] Worker %d executing task %d: %s\n", 
                worker_id, task_id, task->func_name);
    
    // Build argument array
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Execute function pointer directly (for testing without hardware)
    if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    }
}

#endif /* CANN_SDK_AVAILABLE */

// =============================================================================
// Task Completion (Hardware Implementation)
// =============================================================================

#ifdef CANN_SDK_AVAILABLE

void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[A2A3 Core HW] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    // Hardware synchronization using CANN event system
    pthread_mutex_lock(&rt->task_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Advance sliding window
    bool window_advanced = false;
    while (rt->window_oldest_pending < rt->next_task_id) {
        int32_t oldest_slot = PTO_TASK_SLOT(rt->window_oldest_pending);
        if (!rt->pend_task[oldest_slot].is_complete) break;
        rt->window_oldest_pending++;
        window_advanced = true;
    }
    
    DEBUG_PRINT("[A2A3 Core HW] Complete task %d: %s\n", task_id, task->func_name);
    
    // Collect newly ready tasks
    int32_t newly_ready[PTO_MAX_FANOUT];
    int32_t newly_ready_count = 0;
    
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        int32_t dep_slot = PTO_TASK_SLOT(dep_id);
        PendingTask* dep_task = &rt->pend_task[dep_slot];
        
        dep_task->fanin--;
        
        if (dep_task->fanin == 0 && !dep_task->is_complete) {
            newly_ready[newly_ready_count++] = dep_id;
        }
    }
    
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
    
    if (window_advanced) {
        pthread_cond_broadcast(&rt->window_not_full);
    }
    
    pthread_mutex_unlock(&rt->task_mutex);
    
    // Route newly ready tasks to appropriate queues
    for (int i = 0; i < newly_ready_count; i++) {
        a2a3_orch_route_to_queue_threadsafe(rt, newly_ready[i]);
    }
    
    // Signal completion if all tasks done
    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

#else /* No CANN SDK - stub implementation */

void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[A2A3 Core HW STUB] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    pthread_mutex_lock(&rt->task_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    bool window_advanced = false;
    while (rt->window_oldest_pending < rt->next_task_id) {
        int32_t oldest_slot = PTO_TASK_SLOT(rt->window_oldest_pending);
        if (!rt->pend_task[oldest_slot].is_complete) break;
        rt->window_oldest_pending++;
        window_advanced = true;
    }
    
    DEBUG_PRINT("[A2A3 Core HW STUB] Complete task %d: %s\n", task_id, task->func_name);
    
    int32_t newly_ready[PTO_MAX_FANOUT];
    int32_t newly_ready_count = 0;
    
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        int32_t dep_slot = PTO_TASK_SLOT(dep_id);
        PendingTask* dep_task = &rt->pend_task[dep_slot];
        
        dep_task->fanin--;
        
        if (dep_task->fanin == 0 && !dep_task->is_complete) {
            newly_ready[newly_ready_count++] = dep_id;
        }
    }
    
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
    
    if (window_advanced) {
        pthread_cond_broadcast(&rt->window_not_full);
    }
    
    pthread_mutex_unlock(&rt->task_mutex);
    
    for (int i = 0; i < newly_ready_count; i++) {
        a2a3_orch_route_to_queue_threadsafe(rt, newly_ready[i]);
    }
    
    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

#endif /* CANN_SDK_AVAILABLE */

// =============================================================================
// Worker Thread Functions (Hardware Implementation)
// =============================================================================

void* a2a3_vector_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Core HW] Vector worker %d started\n", worker_id);
    
#ifdef CANN_SDK_AVAILABLE
    // Set device context for this worker thread
    aclrtSetDevice(0);
#endif
    
    while (!rt->shutdown_requested) {
        int32_t task_id = a2a3_orch_get_vector_task_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        a2a3_core_execute_task(rt, task_id, worker_id);
        a2a3_core_complete_task(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Core HW] Vector worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

void* a2a3_cube_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Core HW] Cube worker %d started\n", worker_id);
    
#ifdef CANN_SDK_AVAILABLE
    // Set device context for this worker thread
    aclrtSetDevice(0);
#endif
    
    while (!rt->shutdown_requested) {
        int32_t task_id = a2a3_orch_get_cube_task_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        a2a3_core_execute_task(rt, task_id, worker_id);
        a2a3_core_complete_task(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Core HW] Cube worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}
