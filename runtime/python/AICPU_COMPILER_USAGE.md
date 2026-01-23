# AICPU Compiler 使用指南

## 概述

改造后的 CMakeLists.txt 现在完全兼容 `aicpu_compiler.py`，支持灵活的基础平台和客制化代码编译。

## CMakeLists.txt 修改要点

### 1. CMake 变量接收
修改后的 CMakeLists.txt 接收以下三个 CMake 变量（由 aicpu_compiler.py 传入）：

| 变量 | 说明 | 来源 |
|------|------|------|
| `BASE_PLATFORM_PATH` | 基础平台代码路径（如 runtime/aicpu）| 必需，由 aicpu_compiler.py 传入 |
| `CUSTOM_CODE_PATH` | 客制化代码路径（如 runtime/example 或 runtime/graph）| 必需，由 aicpu_compiler.py 传入 |
| `TOOLCHAIN_PATH` | ASCEND 工具链路径 | 可选，若不提供则默认使用环境变量 `ASCEND_HOME_PATH` |

### 2. 源文件自动收集
```cmake
# 自动从基础平台路径递归收集所有 .cpp 文件
file(GLOB_RECURSE BASE_PLATFORM_SOURCES "${BASE_PLATFORM_PATH}/*.cpp")

# 自动从客制化路径递归收集所有 .cpp 文件（包括 graph 等）
file(GLOB_RECURSE CUSTOM_CODE_SOURCES "${CUSTOM_CODE_PATH}/*.cpp")

# 合并：客制化代码优先，可覆盖基础平台实现
set(ALL_SOURCES ${CUSTOM_CODE_SOURCES} ${BASE_PLATFORM_SOURCES})
```

### 3. Include 目录配置
```cmake
target_include_directories(aicpu_graph_kernel
        PRIVATE
            ${BASE_PLATFORM_PATH}              # 基础平台头文件
            ${CUSTOM_CODE_PATH}                # 客制化代码头文件
            ${ASCEND_CANN_PACKAGE_PATH}/include    # ASCEND 头文件
            ...
)
```

## 使用示例

### 使用场景 1：编译基础平台 + 默认例子

```bash
python3 runtime/python/aicpu_compiler.py \
    --base-platform-path runtime/aicpu \
    --custom-code-path runtime/graph \
    --toolchain-path /opt/ascend/toolkit \
    --output-dir ./build/output \
    --cmake-file runtime/aicpu/CMakeLists.txt
```

### 使用场景 2：不指定 TOOLCHAIN_PATH（使用环境变量）

```bash
# 设置环境变量
export ASCEND_HOME_PATH=/opt/ascend/toolkit

# 编译时不指定 --toolchain-path
python3 runtime/python/aicpu_compiler.py \
    --base-platform-path runtime/aicpu \
    --custom-code-path runtime/graph \
    --output-dir ./build/output \
    --cmake-file runtime/aicpu/CMakeLists.txt
```

## 编译流程

当执行 `aicpu_compiler.py` 时，CMakeLists.txt 将：

1. **验证输入**：检查 BASE_PLATFORM_PATH 和 CUSTOM_CODE_PATH 必需变量
2. **配置工具链**：
   - 如果提供了 TOOLCHAIN_PATH，使用该路径
   - 否则使用环境变量 ASCEND_HOME_PATH
3. **收集源文件**：
   - 从 BASE_PLATFORM_PATH 递归收集所有 .cpp 文件
   - 从 CUSTOM_CODE_PATH 递归收集所有 .cpp 文件（包括 graph）
4. **配置编译**：
   - 设置 Include 目录（包括基础平台和客制化路径）
   - 应用编译选项（C++17、优化等）
5. **生成库**：
   - 编译生成 `aicpu_graph_kernel.so` 共享库
   - aicpu_compiler.py 自动将输出复制到指定的 output_dir

## 目录结构示例

```
project/
├── runtime/
│   ├── aicpu/                    # 基础平台代码
│   │   ├── CMakeLists.txt        # 改造后的 CMakeLists.txt
│   │   ├── device_log.cpp
│   │   ├── device_log.h
│   │   └── graph_executor.cpp
│   ├── example/                  # 客制化代码示例
│   │   ├── basic/
│   │   │   ├── graphbuilder.cpp
│   │   │   └── kernels/
│   │   └── basic_python/
│   ├── graph/                    # 图结构代码
│   │   ├── graph.cpp
│   │   └── graph.h
│   └── python/
│       └── aicpu_compiler.py     # 编译器脚本
```

## 关键改进

### 之前（硬编码）
```cmake
# 硬编码源文件列表
target_sources(aicpu_graph_kernel PRIVATE
    graph_executor.cpp
    device_log.cpp
    ../graph/graph.cpp
)

# 硬编码 Include 目录
target_include_directories(aicpu_graph_kernel
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../graph
        ...
)
```

### 之后（动态配置）
```cmake
# 动态收集源文件
file(GLOB_RECURSE BASE_PLATFORM_SOURCES "${BASE_PLATFORM_PATH}/*.cpp")
file(GLOB_RECURSE CUSTOM_CODE_SOURCES "${CUSTOM_CODE_PATH}/*.cpp")
set(ALL_SOURCES ${CUSTOM_CODE_SOURCES} ${BASE_PLATFORM_SOURCES})

# 动态配置 Include 目录
target_include_directories(aicpu_graph_kernel
        PRIVATE ${BASE_PLATFORM_PATH}
        PRIVATE ${CUSTOM_CODE_PATH}
        ...
)
```

## 优势

1. **灵活性**：支持不同的客制化路径（example、graph 等）
2. **可扩展性**：添加新的 .cpp 文件无需修改 CMakeLists.txt
3. **解耦合**：客制化代码和基础平台代码物理分离
4. **覆盖机制**：客制化代码可以覆盖基础平台的实现
5. **环境兼容**：支持环境变量和命令行参数两种方式指定工具链

## 故障排除

### 问题：CMake 配置失败，提示找不到路径

**解决方案**：
- 检查 `--base-platform-path` 和 `--custom-code-path` 是否存在
- 确保路径是绝对路径

### 问题：编译失败，提示找不到头文件

**解决方案**：
- 检查头文件是否存在于指定的路径中
- 确认 Include 路径配置无误

### 问题：生成的 .so 文件不完整

**解决方案**：
- 检查编译日志中的源文件收集输出
- 验证源文件是否在预期位置

