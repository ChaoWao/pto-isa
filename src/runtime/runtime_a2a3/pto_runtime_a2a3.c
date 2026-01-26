/**
 * PTO Runtime System - A2A3 (Ascend) Platform Implementation
 * 
 * A2A3-specific implementations:
 * - Dual ready queue management (vector and cube separation)
 * - Dedicated dependency management module
 * - Heterogeneous worker execution
 * - Task completion with queue-aware dependency propagation
 * 
 * A2A3 Dependency Management:
 * Unlike ARM64's distributed approach (dependency tracking embedded in task_complete),
 * A2A3 uses a dedicated dependency management module that:
 * 1. Routes newly ready tasks to the appropriate queue (vector vs cube)
 * 2. Maintains separate tracking for heterogeneous execution units
 * 3. Supports simulation of the NPU's dual-engine architecture
 */

#include "pto_runtime_a2a3.h"
// Note: pto_runtime_common.h is in parent directory
#include <time.h>

// =============================================================================
// A2A3 Dual Ready Queue Implementation
// =============================================================================

static void vector_ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->vector_ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Vector ready queue overflow\n");
        return;
    }
    
    rt->vector_ready_queue[rt->vector_ready_tail] = task_id;
    rt->vector_ready_tail = (rt->vector_ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count++;
}

static int32_t vector_ready_queue_pop(PTORuntime* rt) {
    if (rt->vector_ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->vector_ready_queue[rt->vector_ready_head];
    rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count--;
    return task_id;
}

static void cube_ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->cube_ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Cube ready queue overflow\n");
        return;
    }
    
    rt->cube_ready_queue[rt->cube_ready_tail] = task_id;
    rt->cube_ready_tail = (rt->cube_ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count++;
}

static int32_t cube_ready_queue_pop(PTORuntime* rt) {
    if (rt->cube_ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->cube_ready_queue[rt->cube_ready_head];
    rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count--;
    return task_id;
}

// =============================================================================
// A2A3 Dedicated Dependency Management Module
// =============================================================================

/**
 * Route a task to the appropriate ready queue based on its is_cube flag.
 * This is the core of A2A3's dedicated dependency management.
 */
static void a2a3_route_to_ready_queue(PTORuntime* rt, int32_t task_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    bool is_cube = rt->pend_task[slot].is_cube;
    
    if (is_cube) {
        cube_ready_queue_push(rt, task_id);
        DEBUG_PRINT("[A2A3 Dep Module] Task %d routed to CUBE queue\n", task_id);
    } else {
        vector_ready_queue_push(rt, task_id);
        DEBUG_PRINT("[A2A3 Dep Module] Task %d routed to VECTOR queue\n", task_id);
    }
}

/**
 * Thread-safe version of routing to ready queue
 */
static void a2a3_route_to_ready_queue_threadsafe(PTORuntime* rt, int32_t task_id) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    bool is_cube = rt->pend_task[slot].is_cube;
    
    if (is_cube) {
        if (rt->cube_ready_count >= PTO_MAX_READY_QUEUE) {
            fprintf(stderr, "[PTO Runtime A2A3] ERROR: Cube ready queue overflow\n");
            pthread_mutex_unlock(&rt->queue_mutex);
            return;
        }
        rt->cube_ready_queue[rt->cube_ready_tail] = task_id;
        rt->cube_ready_tail = (rt->cube_ready_tail + 1) % PTO_MAX_READY_QUEUE;
        rt->cube_ready_count++;
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        DEBUG_PRINT("[A2A3 Dep Module] Task %d routed to CUBE queue (count=%d)\n", task_id, rt->cube_ready_count);
    } else {
        if (rt->vector_ready_count >= PTO_MAX_READY_QUEUE) {
            fprintf(stderr, "[PTO Runtime A2A3] ERROR: Vector ready queue overflow\n");
            pthread_mutex_unlock(&rt->queue_mutex);
            return;
        }
        rt->vector_ready_queue[rt->vector_ready_tail] = task_id;
        rt->vector_ready_tail = (rt->vector_ready_tail + 1) % PTO_MAX_READY_QUEUE;
        rt->vector_ready_count++;
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        DEBUG_PRINT("[A2A3 Dep Module] Task %d routed to VECTOR queue (count=%d)\n", task_id, rt->vector_ready_count);
    }
    
    pthread_mutex_unlock(&rt->queue_mutex);
}

// =============================================================================
// A2A3 Public API Implementation
// =============================================================================

void pto_runtime_enable_a2a3_sim(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers) {
    if (!rt) return;
    rt->simulation_mode = true;
    rt->dual_queue_mode = true;
    rt->num_vector_workers = num_vector_workers;
    rt->num_cube_workers = num_cube_workers;
    // Use dual-mode trace init to distinguish vector (pid=0) and cube (pid=1) workers
    pto_trace_init_dual(num_vector_workers, num_cube_workers);
    DEBUG_PRINT("[PTO Runtime A2A3] Simulation mode enabled: %d vector workers, %d cube workers\n", 
           num_vector_workers, num_cube_workers);
}

int32_t pto_get_ready_task_vector(PTORuntime* rt) {
    return vector_ready_queue_pop(rt);
}

int32_t pto_get_ready_task_cube(PTORuntime* rt) {
    return cube_ready_queue_pop(rt);
}

int32_t pto_get_ready_task_vector_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    while ((rt->vector_ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;
        }
        
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;  // 100µs timeout
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->vector_queue_not_empty, &rt->queue_mutex, &timeout);
        
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->vector_ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->vector_ready_queue[rt->vector_ready_head];
    rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

int32_t pto_get_ready_task_cube_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    while ((rt->cube_ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;
        }
        
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;  // 100µs timeout
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->cube_queue_not_empty, &rt->queue_mutex, &timeout);
        
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->cube_ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->cube_ready_queue[rt->cube_ready_head];
    rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

// =============================================================================
// A2A3 Task Submit Implementation
// =============================================================================

/**
 * A2A3 task submit - routes to appropriate dual queue
 */
void pto_task_submit_a2a3(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
    DEBUG_PRINT("[PTO Runtime A2A3] Submitted task %d: %s (fanin=%d, fanout=%d, is_cube=%d)\n",
           task_id, task->func_name, task->fanin, task->fanout_count, task->is_cube);
    
    // If no dependencies, route to appropriate ready queue
    if (task->fanin == 0) {
        a2a3_route_to_ready_queue_threadsafe(rt, task_id);
        DEBUG_PRINT("[PTO Runtime A2A3] Task %d is ready (no dependencies)\n", task_id);
    }
}

// =============================================================================
// A2A3 Task Complete - Dedicated Dependency Management
// =============================================================================

/**
 * A2A3 task complete with dedicated dependency management module.
 * 
 * Key differences from ARM64:
 * 1. Routes newly ready tasks to cube or vector queue based on is_cube flag
 * 2. Centralized dependency resolution for heterogeneous execution
 */
void pto_task_complete_a2a3(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Advance window_oldest_pending if this was the oldest task
    while (rt->window_oldest_pending < rt->next_task_id) {
        int32_t oldest_slot = PTO_TASK_SLOT(rt->window_oldest_pending);
        if (!rt->pend_task[oldest_slot].is_complete) break;
        rt->window_oldest_pending++;
    }
    
    DEBUG_PRINT("[A2A3 Dep Module] Processing completion of task %d: %s\n", task_id, task->func_name);
    
    // Dedicated dependency management: process all dependents
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        int32_t dep_slot = PTO_TASK_SLOT(dep_id);
        PendingTask* dep_task = &rt->pend_task[dep_slot];
        
        // Update earliest_start_cycle of dependent task
        if (task->end_cycle > dep_task->earliest_start_cycle) {
            dep_task->earliest_start_cycle = task->end_cycle;
        }
        
        dep_task->fanin--;
        DEBUG_PRINT("[A2A3 Dep Module] Task %d fanin decremented to %d\n", 
               dep_id, dep_task->fanin);
        
        if (dep_task->fanin == 0 && !dep_task->is_complete) {
            // Route to appropriate queue via dedicated dependency module
            a2a3_route_to_ready_queue(rt, dep_id);
        }
    }
}

/**
 * Thread-safe A2A3 task complete with dedicated dependency management
 */
void pto_task_complete_a2a3_threadsafe(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    pthread_mutex_lock(&rt->task_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Advance window_oldest_pending if this was the oldest task
    bool window_advanced = false;
    while (rt->window_oldest_pending < rt->next_task_id) {
        int32_t oldest_slot = PTO_TASK_SLOT(rt->window_oldest_pending);
        if (!rt->pend_task[oldest_slot].is_complete) break;
        rt->window_oldest_pending++;
        window_advanced = true;
    }
    
    DEBUG_PRINT("[A2A3 Dep Module] Processing completion of task %d: %s (completed=%lld/%lld)\n", 
           task_id, task->func_name, 
           (long long)rt->total_tasks_completed, 
           (long long)rt->total_tasks_scheduled);
    
    // Collect tasks that become ready
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
    
    // Check if all tasks completed
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
    
    // Signal window_not_full if window advanced
    if (window_advanced) {
        pthread_cond_broadcast(&rt->window_not_full);
    }
    
    pthread_mutex_unlock(&rt->task_mutex);
    
    // Route newly ready tasks via dedicated dependency module
    for (int i = 0; i < newly_ready_count; i++) {
        a2a3_route_to_ready_queue_threadsafe(rt, newly_ready[i]);
    }
    
    // Signal if all tasks are done
    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

// =============================================================================
// A2A3 Task Execution
// =============================================================================

static void execute_task_a2a3(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[Worker A2A3] Executing task %d: %s\n", task_id, task->func_name);
    
    // Build argument array
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Simulation mode
    if (rt->simulation_mode && task->cycle_func) {
        int64_t cycle_cost = task->cycle_func(args, task->num_args);
        
        int64_t worker_current = pto_trace_get_cycle(worker_id);
        
        int64_t actual_start = (worker_current > task->earliest_start_cycle) ? 
            worker_current : task->earliest_start_cycle;
        int64_t actual_end = actual_start + cycle_cost;
        
        task->end_cycle = actual_end;
        
        pto_trace_record_with_time(worker_id, task->func_name, actual_start, actual_end);
        DEBUG_PRINT("[Worker A2A3] Task %d: %s (simulated, %lld cycles, start=%lld)\n", 
               task_id, task->func_name, (long long)cycle_cost, (long long)actual_start);
    }
    // Normal mode
    else if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    }
}

// =============================================================================
// A2A3 Worker Thread Implementation
// =============================================================================

typedef struct {
    PTORuntime* rt;
    int worker_id;
    bool is_cube_worker;
} A2A3WorkerContext;

static void* a2a3_vector_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Vector Worker %d] Started\n", worker_id);
    
    while (!rt->shutdown_requested) {
        int32_t task_id = pto_get_ready_task_vector_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        execute_task_a2a3(rt, task_id, worker_id);
        pto_task_complete_a2a3_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Vector Worker %d] Exiting\n", worker_id);
    free(ctx);
    return NULL;
}

static void* a2a3_cube_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Cube Worker %d] Started\n", worker_id);
    
    while (!rt->shutdown_requested) {
        int32_t task_id = pto_get_ready_task_cube_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        execute_task_a2a3(rt, task_id, worker_id);
        pto_task_complete_a2a3_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Cube Worker %d] Exiting\n", worker_id);
    free(ctx);
    return NULL;
}

// =============================================================================
// A2A3 Runtime Entry Point
// =============================================================================

int runtime_entry_a2a3(PTOOrchFunc orch_func, void* user_data, 
                       int num_vector_workers, int num_cube_workers,
                       int execution_task_threshold) {
    if (!orch_func) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: No orchestration function provided\n");
        return -1;
    }
    
    if (num_vector_workers < 1) num_vector_workers = A2A3_DEFAULT_VECTOR_WORKERS;
    if (num_cube_workers < 1) num_cube_workers = A2A3_DEFAULT_CUBE_WORKERS;
    int total_workers = num_vector_workers + num_cube_workers;
    if (total_workers > PTO_MAX_WORKERS) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Total workers (%d) exceeds maximum (%d)\n",
                total_workers, PTO_MAX_WORKERS);
        return -1;
    }
    if (execution_task_threshold < 0) execution_task_threshold = 0;
    
    printf("[PTO Runtime A2A3] ========================================\n");
    printf("[PTO Runtime A2A3] Heterogeneous Dual-Queue Execution\n");
    printf("[PTO Runtime A2A3] Vector workers: %d\n", num_vector_workers);
    printf("[PTO Runtime A2A3] Cube workers:   %d\n", num_cube_workers);
    if (execution_task_threshold > 0) {
        printf("[PTO Runtime A2A3] Execution threshold: %d tasks (pipelined)\n", execution_task_threshold);
    }
    printf("[PTO Runtime A2A3] ========================================\n");
    
    // Allocate runtime
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "[PTO Runtime A2A3] ERROR: Failed to allocate runtime\n");
        return -1;
    }
    
    // Initialize runtime
    pto_runtime_init(rt);
    pto_runtime_enable_a2a3_sim(rt, num_vector_workers, num_cube_workers);
    rt->num_workers = total_workers;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = execution_task_threshold;
    
    // Spawn vector workers
    printf("[PTO Runtime A2A3] Spawning %d vector workers...\n", num_vector_workers);
    for (int i = 0; i < num_vector_workers; i++) {
        A2A3WorkerContext* ctx = (A2A3WorkerContext*)malloc(sizeof(A2A3WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[PTO Runtime A2A3] ERROR: Failed to allocate worker context\n");
            rt->shutdown_requested = true;
            // Cleanup...
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        ctx->rt = rt;
        ctx->worker_id = i;
        ctx->is_cube_worker = false;
        
        if (pthread_create(&rt->workers[i], NULL, a2a3_vector_worker_func, ctx) != 0) {
            fprintf(stderr, "[PTO Runtime A2A3] ERROR: Failed to create vector worker %d\n", i);
            free(ctx);
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
    }
    
    // Spawn cube workers (worker IDs continue from vector workers)
    printf("[PTO Runtime A2A3] Spawning %d cube workers...\n", num_cube_workers);
    for (int i = 0; i < num_cube_workers; i++) {
        int worker_idx = num_vector_workers + i;
        A2A3WorkerContext* ctx = (A2A3WorkerContext*)malloc(sizeof(A2A3WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[PTO Runtime A2A3] ERROR: Failed to allocate worker context\n");
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        ctx->rt = rt;
        ctx->worker_id = worker_idx;
        ctx->is_cube_worker = true;
        
        if (pthread_create(&rt->workers[worker_idx], NULL, a2a3_cube_worker_func, ctx) != 0) {
            fprintf(stderr, "[PTO Runtime A2A3] ERROR: Failed to create cube worker %d\n", i);
            free(ctx);
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
    }
    
    // Give workers a moment to start
    struct timespec start_delay = {0, 10000000};  // 10ms
    nanosleep(&start_delay, NULL);
    printf("[PTO Runtime A2A3] Workers started, building task graph...\n");
    
    // Build task graph
    orch_func(rt, user_data);
    
    // Mark orchestration complete
    pthread_mutex_lock(&rt->task_mutex);
    rt->execution_started = true;
    int64_t total_tasks = rt->total_tasks_scheduled;
    pthread_mutex_unlock(&rt->task_mutex);
    
    printf("[PTO Runtime A2A3] Task graph built: %lld tasks\n", (long long)total_tasks);
    
    // Wake up workers
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->vector_queue_not_empty);
    pthread_cond_broadcast(&rt->cube_queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for completion
    struct timespec poll_interval = {0, 1000000};  // 1ms
    while (1) {
        pthread_mutex_lock(&rt->task_mutex);
        bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
        int64_t completed = rt->total_tasks_completed;
        pthread_mutex_unlock(&rt->task_mutex);
        
        if (all_done) {
            printf("[PTO Runtime A2A3] All %lld tasks completed!\n", (long long)completed);
            break;
        }
        
        static int64_t last_reported = 0;
        if (completed > last_reported + 1000 || completed == total_tasks) {
            printf("[PTO Runtime A2A3] Progress: %lld / %lld tasks (%.1f%%)\n",
                   (long long)completed, (long long)total_tasks,
                   100.0 * completed / total_tasks);
            last_reported = completed;
        }
        
        nanosleep(&poll_interval, NULL);
    }
    
    // Shutdown
    printf("[PTO Runtime A2A3] Shutting down workers...\n");
    rt->shutdown_requested = true;
    
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->vector_queue_not_empty);
    pthread_cond_broadcast(&rt->cube_queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    for (int i = 0; i < total_workers; i++) {
        pthread_join(rt->workers[i], NULL);
    }
    
    // Statistics
    printf("[PTO Runtime A2A3] ========================================\n");
    printf("[PTO Runtime A2A3] Execution Statistics\n");
    printf("[PTO Runtime A2A3]   Total tasks:     %lld\n", (long long)rt->total_tasks_scheduled);
    printf("[PTO Runtime A2A3]   Completed:       %lld\n", (long long)rt->total_tasks_completed);
    printf("[PTO Runtime A2A3]   Vector workers:  %d\n", num_vector_workers);
    printf("[PTO Runtime A2A3]   Cube workers:    %d\n", num_cube_workers);
    printf("[PTO Runtime A2A3] ========================================\n");
    
    pto_runtime_shutdown(rt);
    free(rt);
    
    return 0;
}
