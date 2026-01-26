#include <cstdint>
#include <atomic>
#include <mutex>
#include "device_log.h"
#include "graph.h"

namespace {

// Shared ready queues for multi-threaded execution (separate by core type)
std::mutex ready_queue_aic_mutex_;
int ready_queue_aic_[GRAPH_MAX_TASKS];
std::atomic<int> ready_count_aic_{0};

std::mutex ready_queue_aiv_mutex_;
int ready_queue_aiv_[GRAPH_MAX_TASKS];
std::atomic<int> ready_count_aiv_{0};

// Task execution tracking
std::atomic<int> completed_tasks_{0};
std::atomic<int> total_tasks_{0};
std::atomic<bool> init_done_{false};

// Thread synchronization for cleanup
std::atomic<int> finished_count_{0};

}  // namespace

/**
 * Execute task graph using polling-based dispatch to AICore
 *
 * This function implements a dynamic task scheduler that:
 * 1. Uses separate shared ready queues for AIC and AIV tasks (protected by mutexes)
 * 2. Each thread polls only its assigned AICore handshake buffers
 * 3. Dispatches ready tasks from matching queue to idle cores based on core type
 * 4. Tracks task completion and updates successor dependencies atomically
 *
 * The scheduler supports arbitrary DAG topologies and automatically handles
 * parallelism across multiple threads and cores based on data dependencies and core types.
 *
 * Algorithm:
 * - Thread 0 initializes shared ready queues, separating tasks by core type
 * - All threads loop while there are tasks ready to run OR tasks executing
 * - Each thread processes only its assigned cores:
 *   - If task completed (idle + task != 0): atomically update dependencies, add to appropriate queue
 *   - If core idle: dispatch from matching queue (AIC core -> AIC queue, AIV core -> AIV queue)
 *
 * @param g Task graph containing all tasks and dependencies
 * @param hank Array of handshake buffers (one per core)
 * @param thread_num Total number of AICPU scheduler threads
 * @param thread_idx Thread identifier (0, 1, 2, ...)
 * @param cur_thread_cores Array of core IDs assigned to this thread
 * @param core_num Number of cores assigned to this thread
 * @return Number of tasks completed by this thread
 */
int execute(Graph& g, Handshake* hank, int thread_num, int thread_idx,
            const int* cur_thread_cores, int core_num) {

    // Thread 0 initializes shared state
    if (thread_idx == 0) {
        DEV_INFO("Thread %d: Initializing graph executor", thread_idx);

        total_tasks_.store(g.get_task_count(), std::memory_order_release);
        completed_tasks_.store(0, std::memory_order_release);

        // Load initial ready tasks and separate by core type
        int initial_ready[GRAPH_MAX_TASKS];
        int initial_count = g.get_initial_ready_tasks(initial_ready);

        DEV_INFO("Thread %d: Found %d initially ready tasks", thread_idx, initial_count);

        int aic_count = 0;
        int aiv_count = 0;
        for (int i = 0; i < initial_count; i++) {
            Task* task = g.get_task(initial_ready[i]);
            if (task->core_type == 0) {  // AIC
                ready_queue_aic_[aic_count++] = initial_ready[i];
                DEV_INFO("Thread %d: Task %d -> AIC queue", thread_idx, initial_ready[i]);
            } else {  // AIV
                ready_queue_aiv_[aiv_count++] = initial_ready[i];
                DEV_INFO("Thread %d: Task %d -> AIV queue", thread_idx, initial_ready[i]);
            }
        }
        ready_count_aic_.store(aic_count, std::memory_order_release);
        ready_count_aiv_.store(aiv_count, std::memory_order_release);

        DEV_INFO("Thread %d: Initial ready tasks: AIC=%d, AIV=%d", thread_idx, aic_count, aiv_count);

        finished_count_.store(0, std::memory_order_release);

        // Signal initialization complete
        init_done_.store(true, std::memory_order_release);
    } else {
        // Other threads wait for initialization
        while (!init_done_.load(std::memory_order_acquire)) {
            // Spin wait
        }
    }

    DEV_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    // Execute tasks using polling-based dispatch
    // Loop until all tasks are completed
    while (completed_tasks_.load(std::memory_order_acquire) < task_count) {

        // Phase 1: Process completed tasks on my managed cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core finished a task (idle + task not null)
            if (h->task_status == 0 && h->task != 0) {
                // Get completed task
                Task* task = reinterpret_cast<Task*>(h->task);
                int task_id = task->task_id;

                DEV_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                // Update fanin of successors atomically and add to appropriate shared ready queue
                for (int j = 0; j < task->fanout_count; j++) {
                    int dep_id = task->fanout[j];
                    Task* dep = g.get_task(dep_id);

                    // Atomic decrement fanin
                    int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                    // Dependency resolved, add to appropriate shared ready queue
                    if (prev_fanin == 1) {
                        if (dep->core_type == 0) {  // AIC task
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            int idx = ready_count_aic_.load(std::memory_order_relaxed);
                            ready_queue_aic_[idx] = dep_id;
                            ready_count_aic_.fetch_add(1, std::memory_order_release);
                            DEV_INFO("Thread %d: Task %d became ready -> AIC queue", thread_idx, dep_id);
                        } else {  // AIV task
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int idx = ready_count_aiv_.load(std::memory_order_relaxed);
                            ready_queue_aiv_[idx] = dep_id;
                            ready_count_aiv_.fetch_add(1, std::memory_order_release);
                            DEV_INFO("Thread %d: Task %d became ready -> AIV queue", thread_idx, dep_id);
                        }
                    }
                }

                // Clear task pointer and update counters
                h->task = 0;
                cur_thread_tasks_in_flight--;
                completed_tasks_.fetch_add(1, std::memory_order_release);
                cur_thread_completed++;
            }
        }

        // Load balancing: Skip dispatch if all my cores are busy
        if (cur_thread_tasks_in_flight >= core_num) {
            continue;
        }

        // Phase 2: Dispatch new tasks from matching ready queue to idle cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core is idle and available (idle + task is null)
            if (h->task_status == 0 && h->task == 0) {
                // Dispatch from matching queue based on core type
                if (h->core_type == 0) {  // AIC core
                    if (ready_count_aic_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                        int count = ready_count_aic_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            ready_count_aic_.fetch_sub(1, std::memory_order_release);
                            int task_id = ready_queue_aic_[count - 1];
                            Task* task = g.get_task(task_id);

                            DEV_INFO("Thread %d: Dispatching AIC task %d to core %d",
                                    thread_idx, task_id, core_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                        }
                    }
                } else if (h->core_type == 1) {  // AIV core
                    if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                        int count = ready_count_aiv_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                            int task_id = ready_queue_aiv_[count - 1];
                            Task* task = g.get_task(task_id);

                            DEV_INFO("Thread %d: Dispatching AIV task %d to core %d",
                                    thread_idx, task_id, core_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                        }
                    }
                }
            }
        }
    }

    DEV_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);

    // Wait for all threads to complete, then reset shared state
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num) {
        // Last thread resets shared state for next execution
        DEV_INFO("Thread %d: Last thread, resetting shared state", thread_idx);
        ready_count_aic_.store(0, std::memory_order_release);
        ready_count_aiv_.store(0, std::memory_order_release);
        completed_tasks_.store(0, std::memory_order_release);
        total_tasks_.store(0, std::memory_order_release);
        init_done_.store(false, std::memory_order_release);
        finished_count_.store(0, std::memory_order_release);
    }

    return cur_thread_completed;
}
