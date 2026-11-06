#!/usr/bin/env bash
# 在 AMD ROCm 机器上安装 PyTorch ROCm 轮子。
#
# 默认使用南京大学 PyTorch wheel 镜像 + ROCm 7.2 / torch 2.14.1。
# 如果机器是 ROCm 6.2 + Python 3.12，可改用：
#   ROCM_INDEX=https://mirrors.nju.edu.cn/pytorch/whl/rocm6.2 \
#   TORCH_SPEC=torch==2.5.1+rocm6.2 bash scripts/setup_rocm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROCM_INDEX="${ROCM_INDEX:-https://mirrors.nju.edu.cn/pytorch/whl/rocm7.2}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.1+rocm7.2}"

if ! command -v uv >/dev/null 2>&1; then
  echo "错误：没有找到 uv，请先安装 https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v rocm-smi >/dev/null 2>&1 && [[ ! -e /dev/kfd ]]; then
  echo "警告：没有检测到 ROCm 驱动（/dev/kfd 或 rocm-smi），仍继续安装。" >&2
fi

echo "[1/3] 安装项目其余依赖，但不让 uv 安装 PyPI 的 torch"
uv sync --no-install-package torch

echo "[2/3] 从镜像安装 ROCm 版 torch：${TORCH_SPEC}"
uv pip install \
  --python .venv/bin/python \
  --index-url "${ROCM_INDEX}" \
  "${TORCH_SPEC}"

echo "[3/3] 验证 ROCm 后端"
.venv/bin/python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda api available:", torch.cuda.is_available())
if torch.version.hip is None:
    raise SystemExit(
        "安装后的 torch 不是 ROCm 构建；请检查 ROCM_INDEX 与 TORCH_SPEC 是否匹配"
    )
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ROCm 版 torch 已安装，但 torch.cuda.is_available() 为 False；请检查 /dev/kfd 权限和驱动")
PY

cat <<'EOF'

ROCm 版 PyTorch 已安装。
注意：以后运行训练时请使用 `uv run --no-sync ...`，避免 uv 重新同步依赖时
把 ROCm torch 替换成 lock 中的 PyPI 版本。
EOF
