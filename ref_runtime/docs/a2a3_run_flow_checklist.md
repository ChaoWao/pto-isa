# a2a3 运行流程检查清单

本文档对照「编译 → 加载 orchestrator/scheduler → 加载 incore function → 拉起 orchestrator/scheduler/worker」四个环节，说明 a2a3 下各步在代码中的位置及如何确认正常。

---

## 根据 log 定位上板问题

**推荐运行方式（便于抓 CANN 日志）：**
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
# 可选：CANN 日志打 stdout
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
cd ref_runtime
./examples/scripts/run_a2a3_once.sh
# 或：python examples/scripts/run_example.py -k examples/easyexample/kernels -g examples/easyexample/golden.py -r rt2 -p a2a3 -v
```

**若运行脚本时终端被杀死（常见为 OOM）：**
- `run_a2a3_once.sh` 已默认设置 `MAKEFLAGS=-j1`，限制 make 单任务编译以降低内存占用。
- 若仍被杀死：先关闭其它占内存程序；或手动 `export MAKEFLAGS=-j1` 后再跑；或在内存更大的环境运行。
- 确认是否 OOM：`dmesg | tail` 或 `journalctl -k -b | grep -i kill` 中可见 Out of memory / Killed process。

**典型失败与 log 对应：**

| 控制台 / 现象 | 含义 | 下一步 |
|---------------|------|--------|
| `Error: rtStreamSynchronize (AICPU) after Init failed: 507018 (Init kernel failed)` | AICPU Init 内核执行失败（SO 加载/入口/参数或路径问题） | 见下文「4) 拉起」及「507018 时对 artifacts 的检查」；查 plog（如 `~/ascend/log/`、`/var/log/ascend_plog/`）中 AICPU 异常详情。控制台会打印 `DeviceRunner: AICPU Init: so_path=...`、`launch_aicpu so_name=...`、`syncing stream...` 及错误码十六进制，便于核对传给 CANN 的 SO 路径与 plog 是否一致。 |
| `DeviceRunner: wrote AICPU SO to /tmp/libaicpu_kernel_<pid>.so` | 当前工作目录不可写，SO 回退到 /tmp | 确保在可写目录运行（如 `cd ref_runtime`），或按 plog 确认设备侧实际加载的 SO 路径 |
| `Registering kernel: func_id=X`、`function_bin_addr=0x...` 正常，随后 507018 | 编译与 kernel 注册正常，问题在 AICPU 侧 Init/Main 执行 | 重点查 AICPU SO 入口符号（`DynTileFwkBackendKernelServerInit`/`DynTileFwkBackendKernelServer` 为 GLOBAL）、CANN 版本与 KFC 接口 |

### 如何生成 plog、如何查 plog

**plog 是什么**
- plog（platform log）是 CANN/昇腾运行时在**运行程序时自动写入**的日志，无需单独“生成”；只要执行了会调设备/rt 的用例（如 `run_example.py -p a2a3`），CANN 就会写 plog。
- 用于排查设备侧错误（如 507018、AICPU 加载 SO 失败、`func is nullptr` 等）。

**如何让 plog 更详细（推荐在跑用例前设置）**
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0    # 0=DEBUG，便于看到 AICPU/KFC 等详细日志
# 可选：把部分 CANN 日志打到当前终端，便于和 plog 对照
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
```
然后正常跑用例，例如：`./examples/scripts/run_a2a3_once.sh` 或 `python examples/scripts/run_example.py ... -p a2a3 -v`。

**plog 常见路径（按优先级看）**
| 路径 | 说明 |
|------|------|
| `$HOME/ascend/log/` | 用户目录下，常见为 `~/ascend/log/` |
| `$ASCEND_HOME/log/` | 若设置了 ASCEND_HOME（如 `/usr/local/Ascend`） |
| `/var/log/ascend_plog/` | 系统级 plog，部分环境写在这里 |
| `$HOME/ascend/log/debug/device-<id>/` | 单设备 debug 日志，如 `device-0/device-<pid>_*.log` |

**怎么查 plog**
1. **按时间**：跑完用例后，看上述目录下**最近修改**的日志文件（`ls -lt` 或 `find ... -mmin -5`）。
2. **按关键字**：在 plog 目录下 grep，例如：
   ```bash
   # 进入 plog 根目录（按你环境选一个）
   cd ~/ascend/log || cd /var/log/ascend_plog || cd $ASCEND_HOME/log

   # 查 507018、Init、AICPU、KFC、so 路径、func is nullptr 等
   grep -r "507018\|0x7bc8a\|Init kernel\|AICPU\|KFC\|libaicpu\|func is nullptr\|DynTile" --include="*.log" .
   ```
3. **看设备侧日志**：若存在 `debug/device-0/`，可直接看该目录下最新的 `device-*_*.log`，里面常有 AICPU 加载/执行错误的直接原因。

**小结**：plog 随运行自动生成；设 `ASCEND_GLOBAL_LOG_LEVEL=0` 后跑用例，再到 `~/ascend/log` 或 `/var/log/ascend_plog` 下按时间/关键字查即可。

**本机 plog 位置确认**：`~/ascend/log` 存在，结构为：
- `~/ascend/log/debug/device-0/`：设备 0 的 device 日志，文件名为 `device-<host_pid>_<timestamp>.log`（如 `device-776507_20260203183924882.log` 对应 host 进程 776507）。
- `~/ascend/log/debug/plog/`：plog 文件，如 `plog-776507_20260203183924673.log`。
- `~/ascend/log/run/`：大量 run 日志。

**507018 根因（从 device-776507 日志）**：device 日志中有：
`Get KFC DynTileFwkBackendKernelServerInit api success, but func is nullptr`，且 `result[11006]`。说明 AICPU 侧按名字找到了 KFC 接口，但**函数指针为空**，即自定义 SO（如 `/tmp/libaicpu_kernel_<pid>.so`）要么未被加载，要么加载后未解析出 `DynTileFwkBackendKernelServerInit`。可能原因：KFC 自定义 SO 需放在 CANN 约定目录（如某配置或白名单路径）而非任意 `/tmp`；或设备侧无法访问 host 的 `/tmp` 路径。需查 CANN 文档中「自定义 AICPU KFC SO 路径」约定。

---

## 1) 编译结果 .o / .so 是否正常生成

### 流程与代码位置

| 产物 | 生成位置 | 验证方式 |
|------|----------|----------|
| **host / aicpu / aicore 三个 binary** | `RuntimeBuilder.build()` → `BinaryCompiler.compile("aicore"|"aicpu"|"host")`，在 `/tmp/{aicore,aicpu,host}_build_*/` 下 CMake 构建，读入内存后目录删除 | 无异常且 `builder.build()` 返回的 `host_binary, aicpu_binary, aicore_binary` 均为非空 bytes |
| **Incore .o（每个 kernel）** | `pto_compiler.compile_incore(source, core_type, pto_isa_root)` → 写 `/tmp/incore_{timestamp}_{pid}.o`，读入后删除 | 无异常且能拿到 `incore_o` bytes，随后 `extract_text_section(incore_o)` 成功 |
| **Orchestration .so** | `pto_compiler.compile_orchestration(source, extra_include_dirs)` → 写 `/tmp/orch_{timestamp}_{pid}.so`，读入后删除 | 无异常且 `orch_so_binary` 非空；host 编排模式下该 SO 仅在 host 侧用于 `run_host_orchestration` 的逻辑，不通过 CANN 加载到设备 |

### 如何确认「正常」

- 控制台出现且无报错：
  - `[1/3] Compiling AICore kernel...` / `[2/3] Compiling AICPU kernel...` / `[3/3] Compiling Host runtime...`
  - `[Incore] Compiling (AIV): ...`（每个 kernel 一条）
  - `Compiled orchestration: XXX bytes`
- 若任一步失败会抛出 `RuntimeError` / `FileNotFoundError`，不会进入后续步骤。

---

## 2) 加载 Orchestrator 与 Scheduler 是否正常

### 概念区分

- **Orchestrator**：负责建图（写 task 列表、依赖等）。rt2 默认 **host orchestration**：在 host CPU 上执行，不把 orch SO 加载到设备。
- **Scheduler（及 AICPU worker）**：在 **libaicpu_kernel.so** 内（AICPU 侧代码），由 CANN 在 `launch_runtime` 时加载并执行。

### 代码路径

**Orchestrator（host 编排）：**

- `runtime.initialize(..., run_orchestrator_on_host=True)` → C API `init_runtime` → `init_runtime_impl()`（`src/runtime/rt2/host/runtime_maker.cpp`）。
- `run_orchestrator_on_host == true` 时：不加载 orch SO，只做 `runtime->set_orch_built_on_host(true)` 和 `set_orch_args()`。
- 建图在 host 上通过 `runtime.run_host_orchestration(host_mirror)` 完成，写 task 到 host 侧 mirror，再拷到设备 PTO2 SM。

**Scheduler（AICPU SO）：**

- `launch_runtime()` → `DeviceRunner::run()` → `ensure_device_initialized()` → `ensure_binaries_loaded()`（`src/platform/a2a3/host/device_runner.cpp`）。
- 其中：
  - 将 `aicpu_so_binary` 写到当前目录 `libaicpu_kernel.so`（供 CANN 按 so_name 加载）；若当前目录不可写则回退到 `/tmp/libaicpu_kernel_<pid>.so`。传给 CANN 的 `so_name` **一律为绝对路径**（`realpath` 规范化，失败时用 `getcwd()` 补全相对路径）。
  - `so_info_.init(aicpu_so_binary, mem_alloc_)` 把 AICPU SO 拷贝到设备 GM。
  - `kernel_args_.init_device_args(device_args_, mem_alloc_)` 初始化设备侧参数。

### 如何确认「正常」

- 控制台应出现：
  - `Host orchestration mode: orchestration will run on host CPU; use allocate_pto2_shared_memory and run_host_orchestration`
  - `DeviceRunner: wrote AICPU SO to <path> for CANN load`
  - `DeviceRunner: binaries loaded`
- 若 AICPU 加载/执行失败，会在 `rtStreamSynchronize(stream_aicpu_)` 报错（如 507018 aicpu exception），见第 4 节。

---

## 3) 加载 Incore Function 是否正常

### 流程与代码位置

- **注册时机**：`set_device()` 之后、`launch_runtime()` 之前；在 `code_runner.py` 里对每个 kernel 执行：
  - `incore_o = pto_compiler.compile_incore(...)` → `kernel_bin = extract_text_section(incore_o)` → `register_kernel(kernel["func_id"], kernel_bin)`。
- **C 侧**：`register_kernel` → `DeviceRunner::register_kernel()`（`device_runner.cpp`）：
  - 要求已 `set_device()`（`stream_aicpu_ != nullptr`）。
  - `mem_alloc_.alloc(alloc_size)` 在设备上分配 GM，把 kernel 的 .text（带 size 头）拷到设备，`func_id_to_addr_[func_id] = function_bin_addr`。
- **使用时机**：`run()` 里对每个 task 设置 `task->function_bin_addr = get_function_bin_addr(task->func_id)`，供 AICPU 侧按 func_id 跳转到对应 incore 代码。

### 如何确认「正常」

- 控制台应出现每个 kernel 的：
  - `Registering kernel: func_id=X, size=XXX bytes`
  - `func_id=X -> function_bin_addr=0x...`
- 在 `=== Setting function_bin_addr for Tasks ===` 阶段，每个 task 的 `function_bin_addr` 应为非 0；若有 `function_bin_addr not found for func_id` 的 Warning，说明该 func_id 未成功注册或未参与建图。

---

## 4) 拉起 Orchestrator、Scheduler、Worker 是否正常

### 顺序（host orchestration）

1. **Host 侧**：`allocate_pto2_shared_memory()` 分配设备侧 SM；`run_host_orchestration(host_mirror)` 在 host 上跑编排，把 task 写入 host mirror，再拷贝到设备 SM。
2. **设备侧**：`launch_runtime()` → `DeviceRunner::run()`：
   - `ensure_device_initialized()`：写 `libaicpu_kernel.so`、AICPU SO 拷到设备、初始化 device args。
   - 设置每个 task 的 `function_bin_addr`（见第 3 节）。
   - `kernel_args_.init_runtime_args(runtime, mem_alloc_)`：Runtime 等结构拷到设备。
   - `launch_aicpu_kernel(..., "DynTileFwkBackendKernelServerInit", 1)`：AICPU 初始化。
   - `rtStreamSynchronize(stream_aicpu_)`：若此处失败（如 **507018 aicpu exception**），说明 AICPU 侧执行异常（SO 加载、入口或参数问题）。
   - `launch_aicpu_kernel(..., "DynTileFwkBackendKernelServer", aicpu_thread_num)`：启动 Scheduler + 3 个 AICPU worker。
   - `launch_aicore_kernel(...)`：启动 AICore 侧 kernel。
   - 再次 `rtStreamSynchronize(stream_aicpu_)`、`rtStreamSynchronize(stream_aicore_)`。

### 如何确认「正常」

- 控制台应依次出现：
  - `DeviceRunner: AICPU Init kernel completed OK`
  - 无 `rtStreamSynchronize (AICPU) failed` / `Aicpu kernel execute failed`
  - 最后 `Launch completed successfully`。
- **常见失败**：`rtStreamSynchronize (AICPU) failed: 507018`、errcode 11006 “inner error”：
  - 表示 AICPU 上执行出错，可能原因：CANN 找不到或无法加载 `libaicpu_kernel.so`（路径/权限）、SO 入口或与当前 CANN 版本不兼容、device args/runtime args 布局与 AICPU 侧预期不一致等。
  - 当前脚本通过 `os.chdir(project_root)` 保证 .so 写在 `ref_runtime/libaicpu_kernel.so`，CANN 会按 so_name 从 host 文件系统解析；需确保该路径可访问且无其它同名 SO 干扰。

---

## 小结

| 检查项 | 关键代码/日志 | 正常标志 |
|--------|----------------|----------|
| 1) .o/.so 生成 | `builder.build()`、`compile_incore`、`compile_orchestration` | 无异常，各 binary 非空，有对应编译日志 |
| 2) Orchestrator/Scheduler 加载 | `init_runtime_impl`（host 编排不加载 orch SO）、`ensure_binaries_loaded` | Host 编排提示、`wrote AICPU SO`、`binaries loaded` |
| 3) Incore 加载 | `register_kernel`、`get_function_bin_addr` | 每个 func_id 有 `function_bin_addr=0x...`，Setting 阶段无 Warning |
| 4) 拉起 orch/scheduler/worker | `launch_aicpu_kernel`（Init + Main）、`launch_aicore_kernel`、`rtStreamSynchronize` | `AICPU Init kernel completed OK`、无 507018、`Launch completed successfully` |

若 1～3 均正常而 4 仍报 507018，重点排查：CANN 对 `libaicpu_kernel.so` 的加载路径与版本兼容性、AICPU 入口符号及与 rt2 的接口约定（args 布局、block_dim、aicpu_num 等）。

### 507018 时对 artifacts 的检查（--keep-artifacts）

可用 `--keep-artifacts DIR` 把编译与加载用到的 .o/.so 保存到目录，便于本地用 `readelf`/`nm` 排查：

```bash
python examples/scripts/run_example.py ... --keep-artifacts build_artifacts
```

**1) 确认 AICPU SO 入口符号存在且为 GLOBAL**

- 路径：`build_artifacts/libaicpu_kernel.so`（或 `ref_runtime/libaicpu_kernel.so`，与运行时写入路径一致）。
- 命令示例：
  - `nm -D build_artifacts/libaicpu_kernel.so | grep DynTile`
  - `readelf -Ws build_artifacts/libaicpu_kernel.so | grep DynTile`
- 预期：能看到 `DynTileFwkBackendKernelServerInit`、`DynTileFwkBackendKernelServer` 为 `FUNC GLOBAL DEFAULT`（T 或 11）。若缺失或为 LOCAL，CANN 无法解析入口会报 11006 / 507018。

**2) 查看 SO 依赖**

- `readelf -d build_artifacts/libaicpu_kernel.so | grep NEEDED`
- AICPU SO 已通过 `-static-libstdc++ -static-libgcc` 静态链接 C++ 运行时和 libgcc，以减少设备侧依赖导致的 11006/507018。若仍失败，可查 CANN 文档确认自定义 AICPU 内核的依赖与 ABI 要求。

**3) 其它可保留的 artifacts**

- `libhost_runtime.so`、`aicore_kernel.bin`、`orchestration.so`、`kernel_func_id_*.o` 会一并写入 `--keep-artifacts` 目录，用于核对编译产物与 incore 符号。

**4) 进一步排查建议**

- 查看设备侧/plog 日志（如 `/var/log/ascend_plog/`、`$ASCEND_HOME/log/` 或 `~/ascend/log/`），确认是否有 SO 加载失败、符号未找到等更具体信息。
- 核对 CANN 版本与 `rtAicpuKernelLaunchExWithArgs`、`KERNEL_TYPE_AICPU_KFC`、`AST_DYN_AICPU` 的兼容性。

**5) 507018 调试结论（当前）**

- 已确认：host 传入的 `so_name`、`kernel_name` 正确（可用 `PTO2_DEBUG_AICPU_LAUNCH=1` 打印）；SO 入口符号存在且 GLOBAL；SO 已静态链接 libstdc++/libgcc，仅依赖 libm、libc。
- CANN 报错中 `soName=`、`funcName=`、`kernelName=` 为空，多为设备侧上报时未回填，不代表 host 未传。
- **plog 根因**：在 `$HOME/ascend/log/debug/device-0/device-<pid>_*.log` 或 `plog/` 下可见：
  - `[AICPU_PROCESSER] Get KFC DynTileFwkBackendKernelServerInit api success, but func is nullptr: (null)`
  - 表示 KFC 层能解析到 kernel 名，但取到的函数指针为 null，即 **加载的 SO 中未解析到该符号**。可能原因：设备侧加载的 SO 不是我们写入的路径（so_name 未被使用或使用了其它 libaicpu_kernel.so）；或 CANN 仅从特定路径加载 SO（例如不加载 `/tmp`）。
- **SO 路径策略**：为排除“CANN 不加载 /tmp”的情况，当前实现 **优先将 AICPU SO 写到当前工作目录** `getcwd()/libaicpu_kernel.so`（与 Python 端 `chdir(project_root)` 一致，通常为 `ref_runtime/libaicpu_kernel.so`），仅当 cwd 不可写时再回退到 `/tmp/libaicpu_kernel_<pid>.so`。传给 CANN 的 `so_name` **始终为绝对路径**（realpath 或 cwd 补全）。若仍 507018，可查看控制台打印的 `wrote AICPU SO to ... (absolute)` 确认实际路径，并对照 CANN 文档确认自定义 AICPU kernel 的 so 路径、kernel 类型（KERNEL_TYPE_AICPU_KFC）、op 名（AST_DYN_AICPU）是否与当前版本一致。
