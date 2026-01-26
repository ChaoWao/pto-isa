/**
 * PTO Runtime - Ascend A2/A3 Simulator Core Worker Implementation
 * 
 * This file implements the worker functions for the simulator platform.
 * Uses pthread synchronization and cycle-accurate simulation.
 */

#include "../../runtime_a2a3/core/a2a3_core_worker.h"
#include "../../runtime_a2a3/orchestration/a2a3_orchestration.h"
#include <stdlib.h>
#include <stdio.h>

// =============================================================================
// Task Execution (Simulator Implementation)
// =============================================================================

void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Core] Worker %d executing task %d: %s\n", 
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
    
    // Simulation mode: use cycle cost function for timing
    if (rt->simulation_mode && task->cycle_func) {
        int64_t cycle_cost = task->cycle_func(args, task->num_args);
        
        int64_t worker_current = pto_trace_get_cycle(worker_id);
        int64_t actual_start = (worker_current > task->earliest_start_cycle) ? 
            worker_current : task->earliest_start_cycle;
        int64_t actual_end = actual_start + cycle_cost;
        
        task->end_cycle = actual_end;
        
        pto_trace_record_with_time(worker_id, task->func_name, actual_start, actual_end);
        DEBUG_PRINT("[A2A3 Core] Task %d simulated: %lld cycles\n", 
                    task_id, (long long)cycle_cost);
    }
    // Also execute actual function if provided (for correctness verification)
    if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    }
}

// =============================================================================
// Task Completion (Simulator Implementation - pthread based)
// =============================================================================

void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[A2A3 Core] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
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
    
    DEBUG_PRINT("[A2A3 Core] Complete task %d: %s\n", task_id, task->func_name);
    
    // Collect newly ready tasks (fanin == 0)
    int32_t newly_ready[PTO_MAX_FANOUT];
    int32_t newly_ready_count = 0;
    
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        int32_t dep_slot = PTO_TASK_SLOT(dep_id);
        PendingTask* dep_task = &rt->pend_task[dep_slot];
        
        dep_task->fanin--;
        
        // Propagate earliest start cycle for cycle-accurate simulation
        if (task->end_cycle > dep_task->earliest_start_cycle) {
            dep_task->earliest_start_cycle = task->end_cycle;
        }
        
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

// =============================================================================
// Worker Thread Functions (Simulator Implementation)
// =============================================================================

void* a2a3_vector_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Core] Vector worker %d started\n", worker_id);
    
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
    
    DEBUG_PRINT("[A2A3 Core] Vector worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

void* a2a3_cube_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Core] Cube worker %d started\n", worker_id);
    
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
    
    DEBUG_PRINT("[A2A3 Core] Cube worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}
