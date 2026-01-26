# ref_runtime 参考文档

> **注意**: `ref_runtime/` 目录包含的是参考项目代码，不属于 pto-isa 项目的一部分。这些代码仅供参考，不应被 pto-isa 项目直接使用。

本文档分析 ref_runtime 的编译流程，以便在 pto-isa 项目中实现类似的编译功能。

## 架构概述

ref_runtime 由三个独立编译的组件组成：

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                        │
│              (example/main.py)                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Host Runtime   │ │   AICPU Kernel   │ │  AICore Kernel   │
│ libhost_runtime  │ │ libaicpu_kernel  │ │  aicore_kernel   │
│     (.so)        │ │     (.so)        │ │     (.o)         │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ 编译器: gcc/g++  │ │ 编译器: aarch64- │ │ 编译器: ccec     │
│ 平台: x86/ARM64  │ │ target-linux-gnu │ │ (Bisheng CCE)    │
│                  │ │ 平台: aarch64    │ │ 平台: AICore     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │   Ascend Device (Hardware)  │
              └────────────────────────────┘
```

## 三个组件详解

### 1. Host Runtime (主机运行时)

**输出**: `libhost_runtime.so`

**编译工具链**:
- 编译器: gcc / g++
- 标准: C++17
- 输出: 共享库 (.so)

**源文件位置**: `src/platform/a2a3/host/`
- `devicerunner.cpp` - 设备管理
- `memoryallocator.cpp` - 内存分配
- `kernel_compiler.cpp` - 内核编译
- `binary_loader.cpp` - 二进制加载
- `pto_runtime_c_api.cpp` - C API 接口

**CMake 编译选项**:
```cmake
target_compile_options(host_runtime PRIVATE
    -Wall -Wextra -std=c++17 -fPIC -O3 -g
)
target_link_libraries(host_runtime PRIVATE
    runtime ascendcl
)
```

### 2. AICPU Kernel (AI CPU 内核)

**输出**: `libaicpu_kernel.so`

**编译工具链**:
- 编译器: `aarch64-target-linux-gnu-gcc/g++` (交叉编译器)
- 位置: `$ASCEND_HOME_PATH/tools/hcc/bin/`
- 标准: C++17
- 输出: 共享库 (.so)

**源文件位置**: `src/platform/a2a3/aicpu/`
- `kernel.cpp` - 内核入口和握手协议
- `device_log.cpp` - 设备日志

**CMake 编译选项**:
```cmake
target_compile_options(aicpu_kernel PRIVATE
    -Wall -Wextra -std=gnu++17 -rdynamic -O3 -fPIC -Werror -g
)
```

### 3. AICore Kernel (AI Core 内核)

**输出**: `aicore_kernel.o`

**编译工具链**:
- 编译器: `ccec` (Bisheng CCE 编译器)
- 链接器: `ld.lld`
- 位置: `$ASCEND_HOME_PATH/bin/`
- 输出: 对象文件 (.o)

**源文件位置**: `src/platform/a2a3/aicore/`
- `kernel.cpp` - 任务执行内核

**编译选项**:
```bash
# AIC (AI Core Cube) 架构
ccec -c -O3 -x cce -std=c++17 --cce-aicore-only \
    --cce-aicore-arch=dav-c220-cube -D__AIC__ \
    -o kernel_aic.o kernel.cpp

# AIV (AI Core Vector) 架构  
ccec -c -O3 -x cce -std=c++17 --cce-aicore-only \
    --cce-aicore-arch=dav-c220-vec -D__AIV__ \
    -o kernel_aiv.o kernel.cpp

# 链接
ld.lld -m aicorelinux -Ttext=0 -static -n \
    -o aicore_kernel.o kernel_aic.o kernel_aiv.o
```

## 编译流程 (main.py 分析)

### 1. BinaryCompiler 初始化

```python
from binary_compiler import BinaryCompiler
compiler = BinaryCompiler()
```

BinaryCompiler 是单例模式，初始化时会：
1. 检查 `ASCEND_HOME_PATH` 环境变量
2. 创建三个工具链对象：
   - `AICoreToolchain` - AICore 编译
   - `AICPUToolchain` - AICPU 编译
   - `HostToolchain` - Host 编译

### 2. 编译 AICore 内核

```python
aicore_include_dirs = [str(runtime_root / "src" / "runtime" / "graph")]
aicore_source_dirs = [str(runtime_root / "src" / "runtime" / "graph")]
aicore_binary = compiler.compile("aicore", aicore_include_dirs, aicore_source_dirs)
```

编译过程：
1. 收集源文件：`src/platform/a2a3/aicore/kernel.cpp` + 自定义源目录
2. 使用 CMake 配置构建
3. 对每个源文件分别编译 AIC 和 AIV 版本
4. 链接所有对象文件生成 `aicore_kernel.o`
5. 读取二进制数据返回

### 3. 编译 AICPU 内核

```python
aicpu_include_dirs = [str(runtime_root / "src" / "runtime" / "graph")]
aicpu_source_dirs = [
    str(runtime_root / "src" / "runtime" / "aicpu"),
    str(runtime_root / "src" / "runtime" / "graph"),
]
aicpu_binary = compiler.compile("aicpu", aicpu_include_dirs, aicpu_source_dirs)
```

编译过程：
1. 收集源文件：`src/platform/a2a3/aicpu/*.cpp` + 自定义源目录
2. 使用 aarch64 交叉编译器
3. 编译为共享库 `libaicpu_kernel.so`
4. 读取二进制数据返回

### 4. 编译 Host 运行时

```python
host_include_dirs = [str(runtime_root / "src" / "runtime" / "graph")]
host_source_dirs = [
    str(runtime_root / "src" / "runtime" / "host"),
    str(runtime_root / "src" / "runtime" / "graph"),
]
host_binary = compiler.compile("host", host_include_dirs, host_source_dirs)
```

编译过程：
1. 收集源文件：固定的 host 源文件 + 自定义源目录
2. 使用标准 gcc/g++ 编译器
3. 链接 CANN 运行时库 (runtime, ascendcl)
4. 编译为共享库 `libhost_runtime.so`
5. 读取二进制数据返回

## 工具链配置

### AICoreToolchain

```python
class AICoreToolchain:
    def __init__(self, cc, ld, aicore_dir, aicore_binary="aicore_kernel.o"):
        self.cc = cc          # ccec 编译器路径
        self.ld = ld          # ld.lld 链接器路径
        self.aicore_dir = aicore_dir
        self.aicore_binary = aicore_binary
    
    def gen_cmake_args(self, include_dirs, source_dirs):
        return " ".join([
            f"-DBISHENG_CC={self.cc}",
            f"-DBISHENG_LD={self.ld}",
            f"-DCUSTOM_INCLUDE_DIRS={';'.join(include_dirs)}",
            f"-DCUSTOM_SOURCE_DIRS={';'.join(source_dirs)}",
        ])
```

### AICPUToolchain

```python
class AICPUToolchain:
    def __init__(self, cc, cxx, aicpu_dir, aicpu_binary="libaicpu_kernel.so", ascend_home_path=None):
        self.cc = cc          # aarch64-target-linux-gnu-gcc
        self.cxx = cxx        # aarch64-target-linux-gnu-g++
        self.aicpu_dir = aicpu_dir
        self.aicpu_binary = aicpu_binary
        self.ascend_home_path = ascend_home_path
    
    def gen_cmake_args(self, include_dirs, source_dirs):
        return " ".join([
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
            f"-DCUSTOM_INCLUDE_DIRS={';'.join(include_dirs)}",
            f"-DCUSTOM_SOURCE_DIRS={';'.join(source_dirs)}",
        ])
```

### HostToolchain

```python
class HostToolchain:
    def __init__(self, cc, cxx, host_dir, binary_name="libhost_runtime.so", ascend_home_path=None):
        self.cc = cc          # gcc
        self.cxx = cxx        # g++
        self.host_dir = host_dir
        self.binary_name = binary_name
        self.ascend_home_path = ascend_home_path
    
    def gen_cmake_args(self, include_dirs, source_dirs):
        return " ".join([
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
            f"-DCUSTOM_INCLUDE_DIRS={';'.join(include_dirs)}",
            f"-DCUSTOM_SOURCE_DIRS={';'.join(source_dirs)}",
        ])
```

## 对 pto-isa 的启示

对于 pto-isa 项目的 `ascend_a2a3` 平台，可以采用类似的编译策略：

### 目标输出结构

```
output/ascend_a2a3/generated_code/
├── orchestration/           # 编排函数 → lib_orchestration.so
│   └── *.c                  # C 源文件
├── incore_aic/              # AI Core Cube 函数 → 编译为 .o 文件
│   └── *.cpp                # C++ 源文件 (PTO ISA API)
└── incore_aiv/              # AI Core Vector 函数 → 编译为 .o 文件
    └── *.cpp                # C++ 源文件 (PTO ISA API)
```

### 编译命令

**Orchestration (编排函数)**:
```bash
gcc -O2 -std=c11 -fPIC -shared -o lib_orchestration.so *.c -lpthread
```

**InCore AIC (AI Core Cube)**:
```bash
ccec -c -O3 -x cce -std=c++17 --cce-aicore-only \
    --cce-aicore-arch=dav-c220-cube -D__AIC__ \
    -o function_aic.o function.cpp
```

**InCore AIV (AI Core Vector)**:
```bash
ccec -c -O3 -x cce -std=c++17 --cce-aicore-only \
    --cce-aicore-arch=dav-c220-vec -D__AIV__ \
    -o function_aiv.o function.cpp
```

### 环境要求

```bash
# 设置 ASCEND 环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## 文件映射

| ref_runtime 组件 | pto-isa 对应 | 说明 |
|-----------------|--------------|------|
| Host Runtime | orchestration/ | 主机端编排代码 |
| AICore Kernel (AIC) | incore_aic/ | AI Core Cube 计算内核 |
| AICore Kernel (AIV) | incore_aiv/ | AI Core Vector 计算内核 |
| AICPU Kernel | (暂不需要) | AICPU 任务调度器 |
