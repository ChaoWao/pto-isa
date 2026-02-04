/**
 * Runtime Class - Implementation
 *
 * Task dependency management with circular ready queue.
 * Follows patterns from pto_runtime.c for consistency.
 */

#include "runtime.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // NOTE: host_api is initialized in InitRuntime() (host-only code)
    // because the CApi functions don't exist when compiled for device.

    // Initialize task array (cannot use memset with atomic members)
    for (int i = 0; i < RUNTIME_MAX_TASKS; i++) {
        tasks[i].task_id = 0;
        tasks[i].func_id = 0;
        tasks[i].num_args = 0;
        tasks[i].function_bin_addr = 0;
        tasks[i].core_type = CoreType::AIV;  // Default to AIV
        tasks[i].fanin = 0;
        tasks[i].fanout_count = 0;
        tasks[i].start_time = 0;
        tasks[i].end_time = 0;
        memset(tasks[i].args, 0, sizeof(tasks[i].args));
        memset(tasks[i].fanout, 0, sizeof(tasks[i].fanout));
    }
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
    block_dim = 0;
    sche_cpu_num = 1;
    tensor_pair_count = 0;
    orch_built_on_host_ = true;
    pto2_gm_sm_ptr_ = nullptr;
    pto2_sm_size_ = 0;
    pto2_gm_heap_ptr_ = nullptr;
    pto2_gm_heap_size_ = 0;
    orch_args_ = nullptr;
    orch_arg_count_ = 0;
    use_pto2_dispatch_ = true;  // default true
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        func_id_to_addr_[i] = 0;
    }
    device_orch_so_data_ = nullptr;
    device_orch_so_size_ = 0;
}

// =============================================================================
// Task Management
// =============================================================================

int Runtime::add_task(uint64_t* args, int num_args, int func_id, CoreType core_type) {
    // Check bounds
    if (next_task_id >= RUNTIME_MAX_TASKS) {
        fprintf(stderr, "[Runtime] ERROR: Task table full (max=%d)\n", RUNTIME_MAX_TASKS);
        return -1;
    }

    if (num_args > RUNTIME_MAX_ARGS) {
        fprintf(stderr, "[Runtime] ERROR: Too many args (%d > %d)\n", num_args, RUNTIME_MAX_ARGS);
        return -1;
    }

    // Allocate task
    int task_id = next_task_id++;
    Task* task = &tasks[task_id];

    // Initialize task fields
    task->task_id = task_id;
    task->func_id = func_id;
    task->num_args = num_args;
    if (args && num_args > 0) {
        memcpy(task->args, args, num_args * sizeof(uint64_t));
    }
    task->function_bin_addr = 0;    // Will be set by host before copying to device
    task->core_type = core_type;    // Set core type
    task->fanin = 0;
    task->fanout_count = 0;
    memset(task->fanout, 0, sizeof(task->fanout));

    return task_id;
}

void Runtime::add_successor(int from_task, int to_task) {
    // Validate task IDs
    if (from_task < 0 || from_task >= next_task_id) {
        fprintf(stderr, "[Runtime] ERROR: Invalid from_task ID %d\n", from_task);
        return;
    }

    if (to_task < 0 || to_task >= next_task_id) {
        fprintf(stderr, "[Runtime] ERROR: Invalid to_task ID %d\n", to_task);
        return;
    }

    Task* from = &tasks[from_task];
    Task* to = &tasks[to_task];

    // Add to_task to from_task's fanout
    if (from->fanout_count >= RUNTIME_MAX_FANOUT) {
        fprintf(stderr, "[Runtime] ERROR: Fanout overflow for task %d (max=%d)\n", from_task, RUNTIME_MAX_FANOUT);
        return;
    }

    from->fanout[from->fanout_count++] = to_task;
    to->fanin++;
}

// =============================================================================
// Query Methods
// =============================================================================

Task* Runtime::get_task(int task_id) {
    if (task_id < 0 || task_id >= next_task_id) {
        return nullptr;
    }
    return &tasks[task_id];
}

int Runtime::get_task_count() const { return next_task_id; }

int Runtime::get_initial_ready_tasks(int* ready_tasks) {
    initial_ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin == 0) {
            initial_ready_tasks[initial_ready_count] = i;
            if (ready_tasks != nullptr) {
                ready_tasks[initial_ready_count] = i;
            }
            initial_ready_count++;
        }
    }
    return initial_ready_count;
}

// =============================================================================
// Utility Methods
// =============================================================================

void Runtime::print_runtime() const {
    printf(
        "\n===================================================================="
        "============\n");
    printf("[Runtime] Task Runtime Status\n");
    printf(
        "======================================================================"
        "==========\n");
    printf("  Total tasks: %d\n", next_task_id);

    // Print initially ready tasks
    printf("\nInitially Ready Tasks (fanin==0):\n");
    printf(
        "----------------------------------------------------------------------"
        "----------\n");
    printf("  ");
    int ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin.load() == 0) {
            if (ready_count > 0) printf(", ");
            printf("%d", i);
            ready_count++;
        }
    }
    if (ready_count == 0) {
        printf("(none)");
    }
    printf("\n  Count: %d\n", ready_count);

    printf("\nTask Table:\n");
    printf(
        "----------------------------------------------------------------------"
        "----------\n");

    for (int i = 0; i < next_task_id; i++) {
        const Task* t = &tasks[i];

        printf("  Task %d: func_id=%d, fanin=%d, fanout=%d, args=%d [",
            i,
            t->func_id,
            t->fanin.load(),
            t->fanout_count,
            t->num_args);

        // Print fanout list
        for (int j = 0; j < t->fanout_count; j++) {
            printf("%d%s", t->fanout[j], j < t->fanout_count - 1 ? "," : "");
        }
        printf("]\n");
    }

    printf(
        "======================================================================"
        "==========\n\n");
}

// =============================================================================
// Tensor Pair Management
// =============================================================================

void Runtime::record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size) {
    if (tensor_pair_count >= RUNTIME_MAX_TENSOR_PAIRS) {
        fprintf(stderr, "[Runtime] ERROR: Tensor pairs full (max=%d)\n", RUNTIME_MAX_TENSOR_PAIRS);
        return;
    }
    tensor_pairs[tensor_pair_count].host_ptr = host_ptr;
    tensor_pairs[tensor_pair_count].dev_ptr = dev_ptr;
    tensor_pairs[tensor_pair_count].size = size;
    tensor_pair_count++;
    printf("Recorded tensor pair: host=%p dev=%p size=%zu\n", host_ptr, dev_ptr, size);
}

TensorPair* Runtime::get_tensor_pairs() {
    return tensor_pairs;
}

int Runtime::get_tensor_pair_count() const {
    return tensor_pair_count;
}

void Runtime::clear_tensor_pairs() {
    tensor_pair_count = 0;
}

// =============================================================================
// Device orchestration
// =============================================================================

bool Runtime::get_orch_built_on_host() const { return orch_built_on_host_; }
void* Runtime::get_pto2_gm_sm_ptr() const { return pto2_gm_sm_ptr_; }
uint64_t* Runtime::get_orch_args() const {
    // Return embedded storage directly (not the pointer) so device code gets correct device address
    return orch_arg_count_ > 0 ? const_cast<uint64_t*>(orch_args_storage_) : nullptr;
}
int Runtime::get_orch_arg_count() const { return orch_arg_count_; }
void Runtime::set_orch_built_on_host(bool v) { orch_built_on_host_ = v; }
void Runtime::set_pto2_gm_sm_ptr(void* p) { pto2_gm_sm_ptr_ = p; }
void Runtime::set_orch_args(uint64_t* args, int count) {
    orch_arg_count_ = count <= RUNTIME_MAX_ARGS ? count : RUNTIME_MAX_ARGS;
    if (args && orch_arg_count_ > 0) {
        memcpy(orch_args_storage_, args, (size_t)orch_arg_count_ * sizeof(uint64_t));
        orch_args_ = orch_args_storage_;
    } else {
        orch_args_ = nullptr;
    }
}

bool Runtime::get_use_pto2_dispatch() const { return use_pto2_dispatch_; }
void Runtime::set_use_pto2_dispatch(bool v) { use_pto2_dispatch_ = v; }

uint64_t Runtime::get_function_bin_addr(int func_id) const {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
    return func_id_to_addr_[func_id];
}
void Runtime::set_function_bin_addr(int func_id, uint64_t addr) {
    if (func_id >= 0 && func_id < RUNTIME_MAX_FUNC_ID) {
        func_id_to_addr_[func_id] = addr;
    }
}
