#!/usr/bin/env bash
# 上板测试：清理后只跑一次 a2a3（rt2），便于根据 log 定位问题
# 使用 ASCEND_GLOBAL_LOG_LEVEL=0 便于 CANN 输出详细日志到 plog/stdout
# 限制 make 并行度（MAKEFLAGS=-j1）降低内存占用，避免 OOM 导致终端被杀死；内存充足时可设 MAKEFLAGS=-j4 等
set -e
REF_RUNTIME="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REF_RUNTIME"
export ASCEND_GLOBAL_LOG_LEVEL=0
# 默认单任务编译，避免 AICore/AICPU 多路并行编译吃满内存被 OOM kill
export MAKEFLAGS="${MAKEFLAGS:--j1}"
# export ASCEND_SLOG_PRINT_TO_STDOUT=1  # 可选：CANN 日志打 stdout
echo "=== ref_runtime: $REF_RUNTIME ==="
echo "=== ASCEND_GLOBAL_LOG_LEVEL=$ASCEND_GLOBAL_LOG_LEVEL ==="
echo "=== MAKEFLAGS=$MAKEFLAGS (降低并行度防 OOM) ==="
echo "=== 清理可能的旧 SO（避免误用）==="
rm -f "$REF_RUNTIME/libaicpu_kernel.so"
ARTIFACTS_DIR="$REF_RUNTIME/build_artifacts"
echo "=== 运行 rt2 + a2a3（--keep-artifacts 保留 SO 便于排查 507018）==="
set +e
python examples/scripts/run_example.py \
  -k examples/easyexample/kernels \
  -g examples/easyexample/golden.py \
  -r rt2 \
  -p a2a3 \
  -v \
  --keep-artifacts build_artifacts
RUN_EXIT=$?
set -e

echo ""
echo "=== 检查 AICPU SO 入口符号（DynTileFwkBackendKernelServerInit / DynTileFwkBackendKernelServer 应为 FUNC GLOBAL）==="
do_nm_readelf() {
  local p="$1"
  [ ! -f "$p" ] && return
  echo "--- $p ---"
  echo "  nm -D:"
  nm -D "$p" 2>/dev/null | grep -E "DynTileFwkBackendKernelServer(Init)?" || true
  echo "  readelf -Ws:"
  readelf -Ws "$p" 2>/dev/null | grep -E "DynTileFwkBackendKernelServer(Init)?" || true
  echo ""
}
do_nm_readelf "$REF_RUNTIME/libaicpu_kernel.so"
do_nm_readelf "$ARTIFACTS_DIR/libaicpu_kernel.so"
if [ ! -f "$REF_RUNTIME/libaicpu_kernel.so" ] && [ ! -f "$ARTIFACTS_DIR/libaicpu_kernel.so" ]; then
  echo "（未找到任一 libaicpu_kernel.so，跳过符号检查；若测试未跑完可能无 SO）"
fi

echo "=== 结束（run_example 退出码: $RUN_EXIT）==="
exit $RUN_EXIT
