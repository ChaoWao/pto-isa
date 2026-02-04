# Rt2 运行时设计文档

本文档描述 ref_runtime 中 **rt2** 运行时的设计，并与仓库根目录下 `docs/runtime_buffer_manager_methods.md`（下称「方法论文档」）做对比。

---

## 1. Rt2 概述

### 1.1 目标与定位

- **Rt2** 是 ref_runtime 中的运行时实现（通过 `run_example.py -r rt2` 选择）。
- 设计目标：
  - 支持 **Orchestrator（编排）→ Scheduler（调度）→ Workers（执行）** 三层架构，与方法论文档中的 PTO Runtime 角色一致。
  - 支持 **设备端编排**：编排函数可在 AICPU 线程上执行（如 thread 3），在设备侧构建任务图并提交到 PTO2 共享内存，Host 仅负责加载运行时、拷贝数据、收尾。
  - 将 **PTO2 相关 C 实现**（原 `src/runtime2`）自包含在 `ref_runtime/src/runtime/rt2/runtime/` 中，不依赖仓库根目录的 `src/runtime2`。

### 1.2 输入与执行流

- **输入**：
  - **编排函数**：控制流 + 任务提交（Host 上或设备上由 `aicpu_orchestration_entry` 等通过 SO 调用执行）。
  - **InCore 内核**：编译为 AICore/AIV 的 kernel，通过 `function_bin_addr` 在 GM 中按统一签名 `void kernel(__gm__ int64_t* args)` 调度。
- **执行流**（设备编排模式）：
  1. Host：分配设备内存、拷贝输入、启动 AICPU 线程（含 1 个 Orchestrator 线程 + 若干 Scheduler 线程）。
  2. Orchestrator（AICPU thread 3）：在设备上构建任务图，通过 `pto2_rt_submit_task` 等写入 PTO2 共享内存（TaskDescriptor、DepList、heap 等）。
  3. Scheduler（AICPU thread 0/1/2）：从共享内存读取任务描述，维护 fanin 就绪、将就绪任务打包为 `PTO2DispatchPayload`，通过 Handshake 下发给 AICore。
  4. Workers（AICore 线程）：从 Handshake 读取 payload，按 `function_bin_addr` 调用 kernel(args)，执行完成后由 AICPU 侧更新完成状态。
  5. Host：等待结束、从 `graph_output_ptr` 拷贝回 Host、释放设备资源。

---

## 2. 目录与模块结构

```
ref_runtime/src/runtime/rt2/
├── build_config.py      # aicore / aicpu / host 的 include_dirs、source_dirs
├── aicore/              # AICore 侧执行与 Handshake
│   └── aicore_executor.cpp
├── aicpu/               # AICPU 侧：Orchestrator 入口 + Scheduler/Worker 逻辑
│   ├── aicpu_executor.cpp
│   ├── device_orchestration_stub.cpp
│   └── example_aicpu_orchestration_entry.cpp
├── host/                # Host 侧：加载 SO、初始化/最终化、拷贝
│   └── runtime_maker.cpp
├── runtime/             # 单一一套运行时源码（C++ Runtime + PTO2 C）
│   ├── runtime.h / runtime.cpp
│   ├── pto2_dispatch_payload.h
│   ├── pto_shared_memory.h/c
│   ├── pto_runtime2*.h/c, pto_scheduler*, pto_worker*, pto_orchestrator*
│   ├── pto_tensormap*, pto_logical_tensor*, pto_interval_tree*
│   ├── pto_ring_buffer*, pto_runtime2_threaded*.h, pto_runtime2_threaded_stub.c
│   └── README.md
└── doc/
    └── rt2_design.md    # 本文档
```

- **build_config.py**：aicpu 的 `include_dirs` / `source_dirs` 仅包含 `runtime`（已合并原 runtime2），无对仓库根 `src/runtime2` 的引用。
- **runtime/**：同时包含「C++ Runtime（Task、Handshake、HostApi）」和「PTO2 C 实现（共享内存、调度器、Worker、Orchestrator、TensorMap、RingBuffer 等）」，便于 ref_runtime 自包含、与 `docs/runtime_buffer_manager_methods.md` 中描述的组件一一对应。

---

## 3. 架构：三层角色与数据流

### 3.1 与文档一致的三层模型

| 角色 | 方法论文档 | Rt2 实现 |
|------|------------|----------|
| **Orchestrator** | 执行编排函数；分配中间缓冲；提交任务；通过 TensorMap 建图；管理 scope | 设备编排：`aicpu_orchestration_entry`（AICPU thread 3）调用 `pto2_rt_submit_task`、`PTO2_SCOPE_BEGIN/END` 等，写入 PTO2 共享内存 |
| **Scheduler** | 维护就绪队列；fanin 解析；将就绪任务分发给 Workers；fanout/缓冲生命周期 | AICPU thread 0/1/2：`resolve_and_dispatch_pto2` 从共享内存读 TaskDescriptor，维护 fanin 计数，构建 `PTO2DispatchPayload`，通过 Handshake 下发 |
| **Workers** | AICore Cube/Vector、AI_CPU、Accelerator 等执行 kernel | AICore：`execute_task_from_payload` 根据 payload 的 `function_bin_addr` 和 `args[]` 调用 kernel |

### 3.2 共享内存布局（PTO2）

- **PTO2SharedMemoryHeader**（见 `pto_shared_memory.h`）：
  - 流控：`current_task_index` / `heap_top`（Orchestrator 写）、`last_task_alive` / `heap_tail`（Scheduler 写）。
  - 布局：`task_descriptors_offset`、`dep_list_pool_offset`、`task_window_size`、`heap_size` 等。
  - 图输出：`graph_output_ptr` / `graph_output_size`（编排侧设置，Host 最终化时从此拷贝回 Host）。
- **TaskDescriptor 环** + **DepListPool** + **Heap** 与文档中「Orchestrator 写、Scheduler 只读」的描述一致；Rt2 中 Scheduler 通过 `PTO2TaskDescriptor`、`PTO2DepListEntry` 解析 fanin/fanout，不修改描述符内容。

### 3.3 调度与下发路径

- **build_pto2_payload**（aicpu_executor.cpp）：根据 `PTO2TaskDescriptor` 和依赖关系，填充 `PTO2DispatchPayload`（task_id、kernel_id、core_type、function_bin_addr、num_args、args[]）。对 `kernel_add_scalar` 等需要标量的 kernel，在 payload 中注入标量（如按 task_id 区分 1.0f / 2.0f）。
- **Handshake**（runtime.h）：每个 AICore 一个 Handshake 槽位；AICPU 写入 `task = &PTO2DispatchPayload`、`task_status = 1`、`core_type`；AICore 读 payload、执行 kernel、写回 `task_status = 0`。
- **AICore**（aicore_executor.cpp）：`execute_task_from_payload` 将 `payload->function_bin_addr` 转为 `UnifiedKernelFunc`，调用 `kernel(payload->args)`，统一签名 `void kernel(__gm__ int64_t* args)`。

---

## 4. 设备编排 vs Host 编排

| 维度 | Host 编排 | 设备编排（Rt2 默认示例） |
|------|-----------|---------------------------|
| 编排函数运行位置 | Host 进程（通过 SO + dlsym 调用） | AICPU thread 3（`aicpu_orchestration_entry`） |
| 图构建位置 | Host 侧 `Runtime::add_task` 等 | 设备侧 `pto2_rt_submit_task`、PTO2 共享内存 |
| Host 职责 | 建图 + 拷贝 + 启动 + 收尾 | 仅拷贝输入、分配 SM、启动线程、拷贝输出、释放 |
| 共享内存 | 可选（Host 建图时可能不用 PTO2 SM） | 必须；Orchestrator 写入，Scheduler 读取 |

- **runtime_maker.cpp**：若 `use_device_orchestration == true`，不加载编排 SO，不调用 Host 侧编排函数；仅设置 `orch_args` 和 PTO2 GM 共享内存指针（由设备侧分配/映射），供 AICPU 线程 3 使用。

---

## 5. 与 docs/runtime_buffer_manager_methods.md 的对比

### 5.1 一致或对应的部分

- **角色划分**：Orchestrator / Scheduler / Workers 三者职责与文档一致；Orchestrator 建图与提交，Scheduler 维护就绪与依赖计数并分发，Workers 执行 InCore kernel。
- **API 概念**：`pto_scope_begin` / `pto_submit_task` / `pto_scope_end` 与文档中的 4 个 API 对应；设备侧通过 `PTO2_SCOPE_BEGIN/END`、`pto2_rt_submit_task` 等 C API 实现。
- **依赖与生命周期**：TaskDescriptor 的 fanin/fanout、DepList 表示的生产者-消费者关系、scope 内 buffer 的 fanout 与释放语义，与文档中「fanout 从 scope_depth 起始」「scope_end 对 [begin_pos, end_pos) 做 fanout 递减」一致。
- **共享内存与只读分工**：文档中「Orchestrator 写、Scheduler 只读」在 Rt2 中通过 PTO2 共享内存布局实现；Scheduler 仅读 TaskDescriptor、DepList、Header，写自己的进度（如 last_task_alive）和完成信号。
- **TensorMap / RingBuffer**：Rt2 的 `pto_tensormap`、`pto_ring_buffer` 对应文档中的 TensorMap 与 ring buffer 池；用于生产者查找与依赖解析。
- **Logical Tensor / Interval Tree**：`pto_logical_tensor`、`pto_interval_tree` 对应文档中的逻辑张量与区间结构，用于张量区域与重叠检测（若启用）。

### 5.2 差异与简化

- **线程模型**：
  - 文档中 Orchestrator 可在 Host 或 Device AICPU 上运行；Rt2 当前实现重点在 **设备编排**（AICPU thread 3 为 Orchestrator）。
  - Rt2 使用 **pto_runtime2_threaded_stub.c**，不链接完整 `pto_runtime2_threaded.c`（无多线程 Orchestrator/Scheduler/Worker 的独立线程池）；设备编排下由 AICPU 上已有的 4 线程（3 Scheduler + 1 Orchestrator）与 AICore 线程协作完成。
- **内核调度方式**：
  - 文档中 InCore 可为「可编程核」或「固定功能加速器」；Rt2 中 AICore 侧统一为「可编程核」：通过 `function_bin_addr` 指向 GM 中的 kernel 二进制，统一签名 `void kernel(__gm__ int64_t* args)`，无显式 opcode/WorkRequestDescriptor。
- **Host API**：
  - 文档中的 `pto_runtime_execute` 等是概念 API；Rt2 的 Host 侧通过 `runtime_maker.cpp` 的 `init_runtime_impl` / `validate_runtime_impl` 与 Python binding（如 `launch_runtime`）对接，设备内存分配/拷贝通过 `Runtime::HostApi`（device_malloc、copy_to_device 等）抽象。
- **数据结构的语言与命名**：
  - 文档中的 Task、BufferMetadata、RefCountState 等是通用描述；Rt2 中对应为 C 的 `PTO2TaskDescriptor`、`PTO2DispatchPayload` 和 C++ 的 `Task`、`Handshake` 等，核心思想一致，实现细节（如 fanin 用原子 refcount 数组）与文档中的「Scheduler 维护 fanin_refcount」一致。

### 5.3 小结对照表

| 主题 | 方法论文档 | Rt2 |
|------|------------|-----|
| 三层角色 | Orchestrator / Scheduler / Workers | 同；设备编排下 Orchestrator 在 AICPU thread 3 |
| 共享内存 | Orchestrator 写、Scheduler 只读 | PTO2SharedMemoryHeader + TaskDescriptor 环 + DepList + Heap |
| 任务描述与下发 | TaskDescriptor → 调度 → Worker | PTO2TaskDescriptor → build_pto2_payload → PTO2DispatchPayload → Handshake → AICore |
| Scope / fanout | scope_begin/end，fanout 从 scope_depth 起 | PTO2_SCOPE_BEGIN/END，pto2 内部 fanout 与 scope 一致 |
| TensorMap / RingBuffer | 生产者查找、ring buffer 池 | pto_tensormap、pto_ring_buffer 在 runtime/ 中 |
| 内核调用方式 | InCore 函数指针或 WorkRequest | 统一 GM 地址 function_bin_addr + args[] |
| 线程/进程 | 文档未绑定具体线程数 | 4 AICPU 线程（3 Scheduler + 1 Orchestrator），多 AICore 线程 |

---

## 6. A2A3 平台编译目标

在 **A2A3 真实硬件**上，代码按运行位置分为两类编译目标，必须严格区分：

| 运行位置 | 内容 | 编译目标 | 说明 |
|----------|------|----------|------|
| **AICPU（设备 CPU）** | 编排函数（Orchestrator）、调度逻辑（Scheduler）、PTO2 运行时 | **ARM64**（aarch64） | 由 `BinaryCompiler` 使用 `aarch64-*-gcc/g++` 交叉编译，产出 `libaicpu_kernel.so`，供 CANN 在设备 AICPU 上加载 |
| **AICore（NPU 核）** | Worker 执行的内核（如 `kernel_add_scalar`、`kernel_mul`） | **CCE Ascend 910B/910C** | 由 `PTOCompiler` 使用 `ccec` 编译，产出 kernel 二进制；通过 `function_bin_addr` 在 GM 中由 AICore 执行 |

- **Orchestrator + Scheduler**：仅包含在 `libaicpu_kernel.so` 中，不包含用户 kernel 源码；构建时通过 rt2 的 `build_config.py` 的 `aicpu.source_dirs`（如 `aicpu`、`runtime`）指定。
- **Worker 内核**：由测试框架（如 `code_runner`）对 `kernels/aiv/*.cpp` 等单独用 `ccec` 编译，将 `.text` 段加载到设备 GM 并注册为 `function_bin_addr`，不在 AICPU SO 中。

---

## 7. 参考

- 仓库根目录：`docs/runtime_buffer_manager_methods.md` — PTO Runtime 缓冲管理与角色划分的详细设计。
- ref_runtime rt2：`runtime/README.md` — runtime 目录下各文件的用途说明。
- 设备编排示例：`aicpu/example_aicpu_orchestration_entry.cpp` — 设备侧编排入口与 `pto2_rt_submit_task` 用法。
