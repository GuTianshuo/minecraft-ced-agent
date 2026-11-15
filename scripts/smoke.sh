#!/usr/bin/env bash
# 三档模型的空转自检。每次改动模型或 schema 后都要跑。
# 用法：scripts/smoke.sh [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -d .venv ] && source .venv/bin/activate

STEPS="${1:-300}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="runs/$STAMP"
mkdir -p "$OUT"

for m in gru tf ced; do
  echo "===== $m ====="
  python train.py --model "$m" --steps "$STEPS" --smoke \
    --ckpt "$OUT/$m.pt" 2>&1 | tee "$OUT/$m.log"
done

echo
echo "logs -> $OUT"
grep -h "^loss" "$OUT"/*.log || true

cat <<EOF

把下面三个数字抄进 docs/decisions/ 里当基线：
  gru 的 tail loss 与成功率为下界
  tf 与 ced 的 tail loss 应该在同等量级，差距 > 20% 先查数据而不是查架构
EOF
