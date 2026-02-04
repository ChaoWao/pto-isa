/**
 * Runtime Builder - Generic Implementation
 *
 * Provides init_runtime_impl and validate_runtime_impl functions that work with
 * pluggable orchestration functions for building task graphs.
 *
 * init_runtime_impl:
 *   - Calls orchestration function to build task graph
 *   - Orchestration is responsible for device memory management
 *
 * validate_runtime_impl (finalize_runtime_impl):
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include "runtime.h"
#include "../runtime/pto_shared_memory.h"
#include <stdint.h>
#include <stddef.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

/**
 * Orchestration function signature.
 *
 * @param runtime   Pointer to Runtime to populate with tasks
 * @param args      Arguments array (host pointers, sizes, etc.)
 * @param arg_count Total number of arguments
 * @return 0 on success, negative on error
 */
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime with dynamic orchestration.
 *
 * This function loads the orchestration SO from binary data via a temp file,
 * resolves the orchestration function via dlsym, then calls it to build the
 * task graph. The orchestration function is responsible for:
 * - Allocating device memory via runtime->host_api.device_malloc()
 * - Copying data to device via runtime->host_api.copy_to_device()
 * - Building the task graph
 * - Recording tensor pairs via runtime->record_tensor_pair()
 *
 * @param runtime           Pointer to pre-constructed Runtime
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function to call
 * @param func_args         Arguments for orchestration (host pointers, sizes, etc.)
 * @param func_args_count   Number of arguments
 * @param use_device_orchestration If true, orchestration runs on AICPU thread 3; do not call orch on host
 * @param run_orchestrator_on_host If true (rt2), orchestration runs on host CPU; caller allocates SM via allocate_pto2_shared_memory
 * @return 0 on success, -1 on failure
 */
int init_runtime_impl(Runtime *runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count,
                    int use_device_orchestration,
                    int run_orchestrator_on_host) {
    // Validate inputs
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }
    if (use_device_orchestration) {
        // Device orchestration: host allocates device memory, copies data,
        // then passes device pointers to AICPU thread 3 via orch_args
        runtime->set_orch_built_on_host(false);

        if (orch_so_binary == nullptr || orch_so_size == 0) {
            std::cerr << "Error: Device orchestration SO is required\n";
            return -1;
        }

        // Copy SO binary to device memory (AICPU cannot access host memory)
        void* dev_so = runtime->host_api.device_malloc(orch_so_size);
        if (!dev_so) {
            std::cerr << "Error: Failed to allocate device memory for orchestration SO\n";
            return -1;
        }
        runtime->host_api.copy_to_device(dev_so, orch_so_binary, orch_so_size);
        runtime->set_device_orch_so(reinterpret_cast<const uint8_t*>(dev_so), orch_so_size);
        // Record for cleanup (no copy-back needed)
        runtime->record_tensor_pair(nullptr, dev_so, orch_so_size);

        std::cout << "Device orchestration mode: SO (" << orch_so_size
                  << " bytes) copied to device memory\n";
        std::cout.flush();

        // Expected args: [host_a, host_b, host_f, size_a, size_b, size_f, SIZE]
        if (func_args_count < 7) {
            std::cerr << "Error: Device orchestration expects at least 7 args, got "
                      << func_args_count << "\n";
            return -1;
        }

        void* host_a = reinterpret_cast<void*>(func_args[0]);
        void* host_b = reinterpret_cast<void*>(func_args[1]);
        void* host_f = reinterpret_cast<void*>(func_args[2]);
        size_t size_a = static_cast<size_t>(func_args[3]);
        size_t size_b = static_cast<size_t>(func_args[4]);
        size_t size_f = static_cast<size_t>(func_args[5]);
        int SIZE = static_cast<int>(func_args[6]);
        size_t BYTES = SIZE * sizeof(float);

        std::cout << "Device orchestration: Allocating device memory...\n";
        std::cout << "  size_a=" << size_a << " size_b=" << size_b << " size_f=" << size_f
                  << " SIZE=" << SIZE << "\n";
        std::cout.flush();

        // Allocate device memory for inputs, outputs, and intermediates
        void* dev_a = runtime->host_api.device_malloc(size_a);
        void* dev_b = runtime->host_api.device_malloc(size_b);
        void* dev_f = runtime->host_api.device_malloc(size_f);
        void* dev_c = runtime->host_api.device_malloc(BYTES);
        void* dev_d = runtime->host_api.device_malloc(BYTES);
        void* dev_e = runtime->host_api.device_malloc(BYTES);

        if (!dev_a || !dev_b || !dev_f || !dev_c || !dev_d || !dev_e) {
            std::cerr << "Error: Failed to allocate device memory for device orchestration\n";
            if (dev_a) runtime->host_api.device_free(dev_a);
            if (dev_b) runtime->host_api.device_free(dev_b);
            if (dev_f) runtime->host_api.device_free(dev_f);
            if (dev_c) runtime->host_api.device_free(dev_c);
            if (dev_d) runtime->host_api.device_free(dev_d);
            if (dev_e) runtime->host_api.device_free(dev_e);
            return -1;
        }

        // Copy input data to device
        runtime->host_api.copy_to_device(dev_a, host_a, size_a);
        runtime->host_api.copy_to_device(dev_b, host_b, size_b);
        std::cout << "  Copied inputs to device\n";

        // Record output tensor for copy-back during finalize
        runtime->record_tensor_pair(host_f, dev_f, size_f);

        // Record device-only allocations for cleanup (no copy-back)
        runtime->record_tensor_pair(nullptr, dev_a, size_a);
        runtime->record_tensor_pair(nullptr, dev_b, size_b);
        runtime->record_tensor_pair(nullptr, dev_c, BYTES);
        runtime->record_tensor_pair(nullptr, dev_d, BYTES);
        runtime->record_tensor_pair(nullptr, dev_e, BYTES);

        // Allocate GM heap for orchestrator output buffers (must persist until tasks complete)
        // Need enough space for all task outputs; 4 tasks * 65536 bytes each = 256KB minimum
        // Use 512KB for safety margin
        const size_t GM_HEAP_SIZE = 512 * 1024;  // 512KB
        void* gm_heap = runtime->host_api.device_malloc(GM_HEAP_SIZE);
        if (!gm_heap) {
            std::cerr << "Error: Failed to allocate GM heap for device orchestration\n";
            // Note: already recorded tensor pairs will be freed in validate_runtime_impl
            return -1;
        }
        runtime->record_tensor_pair(nullptr, gm_heap, GM_HEAP_SIZE);

        // Build new args with device pointers
        // Layout: [dev_a, dev_b, dev_f, size_a, size_b, size_f, SIZE, dev_c, dev_d, dev_e, gm_heap, heap_size]
        static uint64_t device_args[12];  // Static to persist beyond function scope
        device_args[0] = reinterpret_cast<uint64_t>(dev_a);
        device_args[1] = reinterpret_cast<uint64_t>(dev_b);
        device_args[2] = reinterpret_cast<uint64_t>(dev_f);
        device_args[3] = size_a;
        device_args[4] = size_b;
        device_args[5] = size_f;
        device_args[6] = SIZE;
        device_args[7] = reinterpret_cast<uint64_t>(dev_c);
        device_args[8] = reinterpret_cast<uint64_t>(dev_d);
        device_args[9] = reinterpret_cast<uint64_t>(dev_e);
        device_args[10] = reinterpret_cast<uint64_t>(gm_heap);
        device_args[11] = GM_HEAP_SIZE;

        runtime->set_orch_args(device_args, 12);
        runtime->set_pto2_gm_sm_ptr(nullptr);  // Caller sets via set_pto2_gm_sm_ptr after allocating SM

        std::cout << "Device orchestration: Ready for AICPU execution\n";
        return 0;
    }
    if (run_orchestrator_on_host) {
        // Host orchestration (rt2): do not load orch SO; host will call host_orchestration_entry after allocate_pto2_shared_memory
        runtime->set_orch_built_on_host(true);  // so AICPU executor reads task count from SM and skips waiting for thread 3
        runtime->set_orch_args(func_args, func_args_count);
        std::cout << "Host orchestration mode: orchestration will run on host CPU; use allocate_pto2_shared_memory and run_host_orchestration\n";
        return 0;
    }

    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name == nullptr) {
        std::cerr << "Error: Invalid orchestration parameters\n";
        return -1;
    }

    runtime->set_orch_built_on_host(true);

    // Load orchestration SO from binary data via temp file
    char fd_path[128];
    snprintf(fd_path, sizeof(fd_path), "/tmp/orch_so_%d.so", getpid());

    int fd = open(fd_path, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd < 0) {
        std::cerr << "Error: Failed to create temp SO file\n";
        return -1;
    }

    ssize_t written = write(fd, orch_so_binary, orch_so_size);
    if (written < 0 || static_cast<size_t>(written) != orch_so_size) {
        std::cerr << "Error: Failed to write orchestration SO to temp file\n";
        close(fd);
        unlink(fd_path);
        return -1;
    }
    close(fd);

    void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    unlink(fd_path);
    if (handle == nullptr) {
        std::cerr << "Error: dlopen failed: " << dlerror() << "\n";
        return -1;
    }

    dlerror();  // Clear any existing error
    OrchestrationFunc orch_func =
        reinterpret_cast<OrchestrationFunc>(dlsym(handle, orch_func_name));
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        std::cerr << "Error: dlsym failed for '" << orch_func_name << "': " << dlsym_error << "\n";
        dlclose(handle);
        return -1;
    }

    std::cout << "Loaded orchestration function: " << orch_func_name << "\n";

    // Clear any previous tensor pairs
    runtime->clear_tensor_pairs();

    std::cout << "\n=== Calling Orchestration Function ===" << '\n';
    std::cout << "Args count: " << func_args_count << '\n';

    // Call orchestration function to build task graph
    // The orchestration function handles device memory allocation and copy-to-device
    int rc = orch_func(runtime, func_args, func_args_count);
    if (rc != 0) {
        std::cerr << "Error: Orchestration function failed with code " << rc << '\n';
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    std::cout << "\nRuntime initialized. Ready for execution from Python.\n";

    // Note: We intentionally leak the dlopen handle to keep the SO loaded
    // for the lifetime of the process.

    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Frees device memory for recorded tensors
 * 3. Clears tensor pair state
 *
 * @param runtime  Pointer to Runtime
 * @return 0 on success, -1 on failure
 */
int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    int rc = 0;

    std::cout << "\n=== Copying Results Back to Host ===" << '\n';

    // Copy all recorded tensors from device back to host
    TensorPair* tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    std::cout << "Tensor pairs to copy: " << tensor_pair_count << '\n';

    // Validate host_api is properly initialized
    if (runtime->host_api.copy_from_device == nullptr) {
        std::cerr << "Error: host_api.copy_from_device not initialized\n";
        return -1;
    }

    // PTO2 (device or host orchestration): graph output may be in packed buffer; copy from graph_output_ptr when set
    // Note: pto2_sm is a DEVICE pointer, must copy header to host first before reading
    void* pto2_sm = runtime->get_pto2_gm_sm_ptr();
    uint64_t graph_out_ptr = 0;
    int32_t graph_out_size = 0;
    if (pto2_sm != nullptr) {
        // Copy header from device to host to read graph_output_ptr/size
        PTO2SharedMemoryHeader host_header;
        int hdr_rc = runtime->host_api.copy_from_device(&host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
        if (hdr_rc == 0) {
            graph_out_ptr = host_header.graph_output_ptr;
            graph_out_size = host_header.graph_output_size;
        } else {
            std::cerr << "Warning: Failed to copy PTO2 header from device, using default tensor pairs\n";
        }
    }

    bool first_output_tensor = true;  // Track first tensor with host_ptr (the actual output)
    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];

        // Skip if device pointer is null (nothing to copy or free)
        if (pair.dev_ptr == nullptr) {
            std::cerr << "Warning: Tensor " << i << " has null device pointer, skipping\n";
            continue;
        }

        // If host pointer is null, this is a device-only allocation (e.g., gm_heap)
        // Will be freed but no copy-back needed
        if (pair.host_ptr == nullptr) {
            std::cout << "Tensor " << i << ": device-only allocation (no copy-back)\n";
            continue;
        }

        void* src_ptr = pair.dev_ptr;
        size_t copy_size = pair.size;
        // Use graph_output_ptr for the first tensor that has a host pointer (the actual output)
        if (first_output_tensor && graph_out_ptr != 0 && graph_out_size > 0) {
            src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(graph_out_ptr));
            copy_size = static_cast<size_t>(graph_out_size);
            std::cout << "Using packed output buffer: ptr=0x" << std::hex << graph_out_ptr
                      << std::dec << ", size=" << graph_out_size << '\n';
            first_output_tensor = false;
        }

        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, src_ptr, copy_size);
        if (copy_rc != 0) {
            std::cerr << "Error: Failed to copy tensor " << i << " from device: " << copy_rc << '\n';
            rc = copy_rc;
            // Continue with cleanup anyway
        } else {
            std::cout << "Tensor " << i << ": " << pair.size << " bytes copied to host\n";
        }
    }

    // Note: PrintHandshakeResults is now called in DeviceRunner's destructor

    // Cleanup device tensors
    if (runtime->host_api.device_free != nullptr && tensor_pair_count > 0) {
        std::cout << "\n=== Cleaning Up ===" << '\n';
        for (int i = 0; i < tensor_pair_count; i++) {
            if (tensor_pairs[i].dev_ptr != nullptr) {
                runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
            }
        }
        std::cout << "Freed " << tensor_pair_count << " device tensors\n";
    }

    // Clear tensor pairs
    runtime->clear_tensor_pairs();

    std::cout << "=== Finalize Complete ===" << '\n';

    return rc;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
