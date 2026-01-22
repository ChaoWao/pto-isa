/**
 * AICPU Kernel Launcher Example
 * This program demonstrates how to launch an AICPU kernel using CANN runtime
 * APIs
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <runtime/rt.h>
#include <string>
#include <vector>
#include "graph/graph.h"
#include <assert.h>

/**
 * @brief Kernel arguments for the AICPU kernel
 * @details This structure is used to pass arguments to the AICPU kernel.
 * It contains a pointer to the device arguments.
 * The thing important is
 *   1. the offset from KernelArgs to deviceArgs pointer
 *   2. the offset from DeviceArgs to aicpuSoBin
 *   3. the offset from DeviceArgs to aicpuSoLen
 * which are a hardcoded in the AICPU kernel("libaicpu_extend_kernels.so").
 * There are also hardcoded three function names(see hello_world.cpp):
 *   1. StaticTileFwkBackendKernelServer
 *   2. DynTileFwkBackendKernelServerInit
 *   3. DynTileFwkBackendKernelServer
 * which will called when you launch the following kernels of
 * libtilefwk_backend_server.so:
 *   1. StaticTileFwkKernelServer
 *   2. DynTileFwkKernelServerInit
 *   3. DynTileFwkKernelServer
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
};

static int ReadFile(const std::string &path, std::vector<uint8_t> &buf) {//以二进制方式完整读取一个文件的内容到buf
    std::ifstream fs(path, std::ios::binary | std::ios::ate);
    if (!fs.is_open()) {
        std::cerr << "无法打开内核文件: " << path << '\n';
        return -1;
    }
    std::streamsize size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(size));
    if (!fs.read(reinterpret_cast<char *>(buf.data()), size)) {
        std::cerr << "读取内核文件失败: " << path << '\n';
        return -1;
    }
    return 0;
}

struct KernelArgs {
    uint64_t unused[5] = {0};
    int64_t *deviceArgs{nullptr};
    int64_t *hankArgs{nullptr};
    int64_t core_num;
    Graph *graphArgs{nullptr};

    int InitDeviceArgs(const DeviceArgs &hostDeviceArgs) {
        // Allocate device memory for deviceArgs
        if (deviceArgs == nullptr) {
            void *deviceArgsDev = nullptr;
            uint64_t deviceArgsSize = sizeof(DeviceArgs);
            int allocRc = rtMalloc(&deviceArgsDev, deviceArgsSize, RT_MEMORY_HBM, 0);
            if (allocRc != 0) {
                std::cerr << "Error: rtMalloc for deviceArgs failed: " << allocRc << '\n';
                return allocRc;
            }
            deviceArgs = reinterpret_cast<int64_t *>(deviceArgsDev);
        }
        // Copy hostDeviceArgs to device memory via deviceArgs
        int rc =
            rtMemcpy(deviceArgs, sizeof(DeviceArgs), &hostDeviceArgs, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
            rtFree(deviceArgs);
            deviceArgs = nullptr;
            return rc;
        }
        return 0;
    }

    int FinalizeDeviceArgs() {
        if (deviceArgs != nullptr) {
            int rc = rtFree(deviceArgs);
            deviceArgs = nullptr;
            return rc;
        }
        return 0;
    }

    int InitGraphArgs(const Graph& hostGraph) {
        if (graphArgs == nullptr) {
            void *graphDev = nullptr;
            uint64_t graphSize = sizeof(Graph);
            int allocRc = rtMalloc(&graphDev, graphSize, RT_MEMORY_HBM, 0);
            if (allocRc != 0) {
                std::cerr << "Error: rtMalloc for graphArgs failed: " << allocRc << '\n';
                return allocRc;
            }
            graphArgs = reinterpret_cast<Graph*>(graphDev);
        }
        int rc = rtMemcpy(graphArgs, sizeof(Graph), &hostGraph, sizeof(Graph), RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy for graph failed: " << rc << '\n';
            rtFree(graphArgs);
            graphArgs = nullptr;
            return rc;
        }
        return 0;
    }

    int FinalizeGraphArgs() {
        if (graphArgs != nullptr) {
            int rc = rtFree(graphArgs);
            graphArgs = nullptr;
            return rc;
        }
        return 0;
    }
};

struct AicpuSoInfo {
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};

    int Init(const std::string &soPath) {
        std::ifstream file(soPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << soPath << '\n';
            return -1;
        }

        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);

        void *dAicpuData = nullptr;
        // Don't know why use 0 for module id, but it works.
        int rc = rtMalloc(&dAicpuData, fileSize, RT_MEMORY_HBM, 0);
        if (rc != 0) {
            std::cerr << "Error: rtMalloc failed: " << rc << '\n';
            return rc;
        }
        rc = rtMemcpy(dAicpuData, fileSize, buffer.data(), fileSize, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
            rtFree(dAicpuData);
            dAicpuData = nullptr;
            return rc;
        }

        aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
        aicpuSoLen = fileSize;
        return 0;
    }

    int Finalize() {
        if (aicpuSoBin != 0) {
            int rc = rtFree(reinterpret_cast<void *>(aicpuSoBin));
            aicpuSoBin = 0;
            return rc;
        }
        return 0;
    }
};

struct Handshake {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile int32_t control;  // 0=execute, 1=quit
    volatile int32_t task;     // task ID: -1=none, 0=TADD, etc.
} __attribute__((aligned(64)));

class DeviceRunner {
  public:
    static DeviceRunner &Get() {
        static DeviceRunner runner;
        return runner;
    }

    int LaunchAiCpuKernel(rtStream_t stream, KernelArgs *kArgs, const char *kernelName, int aicpuNum) {
        struct Args {
            KernelArgs kArgs;
            char kernelName[32];
            const char soName[32] = {"libaicpu_extend_kernels.so"};
            const char opName[32] = {""};
        } args;

        args.kArgs = *kArgs;
        std::strncpy(args.kernelName, kernelName, sizeof(args.kernelName) - 1);
        args.kernelName[sizeof(args.kernelName) - 1] = '\0';

        rtAicpuArgsEx_t rtArgs;
        std::memset(&rtArgs, 0, sizeof(rtArgs));
        rtArgs.args = &args;
        rtArgs.argsSize = sizeof(args);
        rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
        rtArgs.soNameAddrOffset = offsetof(struct Args, soName);

        return rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, &rtArgs,
                                             nullptr, stream, 0);
    }

    int LauncherAicoreKernel(rtStream_t stream, KernelArgs *kernelArgs) {
        const std::string binPath = "./aicore/kernel.o";
        std::vector<uint8_t> bin;
        if (ReadFile(binPath, bin) != 0) {
            return -1;
        }

        size_t binSize = bin.size();
        const void *binData = bin.data();

        rtDevBinary_t binary;
        std::memset(&binary, 0, sizeof(binary));
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;
        binary.data = binData;
        binary.length = binSize;
        void *binHandle = nullptr;
        int rc = rtRegisterAllKernel(&binary, &binHandle);
        if (rc != RT_ERROR_NONE) {
            std::cerr << "rtRegisterAllKernel失败: " << rc << '\n';
            return rc;
        }

        struct Args {
            int64_t *hankArgs;
        };
        Args args = {kernelArgs->hankArgs};
        rtArgsEx_t rtArgs;
        std::memset(&rtArgs, 0, sizeof(rtArgs));
        rtArgs.args = &args;
        rtArgs.argsSize = sizeof(args);

        rtTaskCfgInfo_t cfg = {};
        cfg.schemMode = RT_SCHEM_MODE_BATCH;

        rc = rtKernelLaunchWithHandleV2(binHandle, 0, 1, &rtArgs, nullptr, stream, &cfg);
        if (rc != RT_ERROR_NONE) {
            std::cerr << "rtKernelLaunchWithHandleV2失败: " << rc << '\n';
            return rc;
        }

        return rc;
    }

    int Run(rtStream_t streamAicpu, rtStream_t streamAicore, KernelArgs *kernelArgs, int launchAicpuNum = 5) {
        if (kernelArgs == nullptr) {
            std::cerr << "Error: kernelArgs is null" << '\n';
            return -1;
        }

        // Launch init which save the Aicpu So to device and bind the function names
        int rc = LaunchAiCpuKernel(streamAicpu, kernelArgs, "DynTileFwkKernelServerInit", 1);
        if (rc != 0) {
            std::cerr << "Error: LaunchAiCpuKernel (init) failed: " << rc << '\n';
            return rc;
        }

        // Launch main kernel
        rc = LaunchAiCpuKernel(streamAicpu, kernelArgs, "DynTileFwkKernelServer", launchAicpuNum);
        if (rc != 0) {
            std::cerr << "Error: LaunchAiCpuKernel (main) failed: " << rc << '\n';
            return rc;
        }

        rc = LauncherAicoreKernel(streamAicore, kernelArgs);

        return 0;
    }

  private:
    DeviceRunner() {}
};

void MvHankArg(KernelArgs& kernelArgs, Handshake* args, int num_cores) {

    void *hankDev = nullptr;
    size_t total_size = sizeof(Handshake) * num_cores;
    int rc = rtMalloc(&hankDev, total_size, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        std::cerr << "Error: rtMalloc failed: " << rc << '\n';
        rtFree(hankDev);
        hankDev = nullptr;
        return;
    }

    rc = rtMemcpy(hankDev, total_size, args, total_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
        rtFree(hankDev);
        hankDev = nullptr;
        return;
    }

    kernelArgs.hankArgs = reinterpret_cast<int64_t *>(hankDev);
    kernelArgs.core_num = num_cores;
}


void PrintResult(KernelArgs& kernelArgs, int num_cores) {
    std::vector<Handshake> host_results(num_cores);
    size_t total_size = sizeof(Handshake) * num_cores;
    rtMemcpy(host_results.data(), total_size, kernelArgs.hankArgs, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Handshake results for " << num_cores << " cores:" << std::endl;
    for (int i = 0; i < num_cores; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << host_results[i].aicore_done
                  << " aicpu_ready=" << host_results[i].aicpu_ready
                  << " control=" << host_results[i].control
                  << " task=" << host_results[i].task << std::endl;
    }
}


// Example usage
int main(int argc, char **argv) {
    std::cout << "=== Launching AICPU Kernel with Graph ===" << '\n';

    constexpr int NUM_CORES = 3;  // 1 AIC + 2 AIV cores for 1c2v architecture
    Handshake hankArgs[NUM_CORES] = {};
    // Initialize all cores' handshake buffers
    for (int i = 0; i < NUM_CORES; i++) {
        hankArgs[i].aicpu_ready = 0;
        hankArgs[i].aicore_done = 0;
        hankArgs[i].control = 0;  // 0=execute
        hankArgs[i].task = -1;    // -1=no task initially
    }

    // Parse device id from main's argument (expected range: 0-15)
    int deviceId = 9;
    if (argc > 1) {
        try {
            deviceId = std::stoi(argv[1]);
            if (deviceId < 0 || deviceId > 15) {
                std::cerr << "Error: deviceId (" << deviceId << ") out of range [0, 15]" << '\n';
                return -1;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error: invalid deviceId argument: " << argv[1] << '\n';
            return -1;
        }
    }
    int devRc = rtSetDevice(deviceId);
    if (devRc != 0) {
        std::cerr << "Error: rtSetDevice(" << deviceId << ") failed: " << devRc << '\n';
        return devRc;
    }

    rtStream_t streamAicpu = nullptr;
    rtStream_t streamAicore = nullptr;
    int rc = rtStreamCreate(&streamAicpu, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate failed: " << rc << '\n';
        return rc;
    }

    rc = rtStreamCreate(&streamAicore, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate failed: " << rc << '\n';
        return rc;
    }

    std::string soPath = "./aicpu/libaicpu_graph_kernel.so";
    AicpuSoInfo soInfo{};
    rc = soInfo.Init(soPath);
    if (rc != 0) {
        std::cerr << "Error: AicpuSoInfo::Init failed: " << rc << '\n';
        rtStreamDestroy(streamAicpu);
        rtStreamDestroy(streamAicore);
        streamAicpu = nullptr;
        streamAicore = nullptr;
        return rc;
    }

    KernelArgs kernelArgs{};
    DeviceArgs deviceArgs{};
    deviceArgs.aicpuSoBin = soInfo.aicpuSoBin;
    deviceArgs.aicpuSoLen = soInfo.aicpuSoLen;
    rc = kernelArgs.InitDeviceArgs(deviceArgs);

    MvHankArg(kernelArgs, hankArgs, NUM_CORES);

    if (rc != 0) {
        std::cerr << "Error: KernelArgs::InitDeviceArgs failed: " << rc << '\n';
        soInfo.Finalize();
        rtStreamDestroy(streamAicpu);
        rtStreamDestroy(streamAicore);
        streamAicpu = nullptr;
        streamAicore = nullptr;
        return rc;
    }

    // Create a test graph to pass to the kernel
    std::cout << "\n=== Creating Test Graph for Kernel ===" << '\n';
    Graph testGraph;
    uint64_t args[] = {1, 2, 3};
    int t0 = testGraph.add_task(args, 3, 0);
    int t1 = testGraph.add_task(args, 3, 1);
    int t2 = testGraph.add_task(args, 3, 2);
    testGraph.add_successor(t0, t1);
    testGraph.add_successor(t1, t2);
    std::cout << "Created graph with " << testGraph.get_task_count() << " tasks in a pipeline\n";
    testGraph.print_graph();

    // Initialize graph args
    rc = kernelArgs.InitGraphArgs(testGraph);
    if (rc != 0) {
        std::cerr << "Error: KernelArgs::InitGraphArgs failed: " << rc << '\n';
        kernelArgs.FinalizeDeviceArgs();
        soInfo.Finalize();
        rtStreamDestroy(streamAicpu);
        rtStreamDestroy(streamAicore);
        streamAicpu = nullptr;
        streamAicore = nullptr;
        return rc;
    }
    std::cout << "Graph transferred to device memory\n\n";

    DeviceRunner &runner = DeviceRunner::Get();
    int launchAicpuNum = 1;
    rc = runner.Run(streamAicpu, streamAicore, &kernelArgs, launchAicpuNum);

    rc = rtStreamSynchronize(streamAicpu);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize failed: " << rc << '\n';
        return rc;
    }

    rc = rtStreamSynchronize(streamAicore);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize failed: " << rc << '\n';
        return rc;
    }

    PrintResult(kernelArgs, NUM_CORES);
    kernelArgs.FinalizeGraphArgs();
    kernelArgs.FinalizeDeviceArgs();
    soInfo.Finalize();
    rtStreamDestroy(streamAicpu);
    rtStreamDestroy(streamAicore);
    streamAicpu = nullptr;
    streamAicore = nullptr;

    if (rc != 0) {
        std::cerr << "=== Launch Failed ===" << '\n';
    } else {
        std::cout << "=== Launch Success ===" << '\n';
    }

    return rc;
}
