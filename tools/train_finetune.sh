#!/usr/bin/env bash
# 说明：HRSC 不大，上述 36 epoch 已足够；如果 val Recall 未达标，可把 EPOCHS 提到 48，再观察
set -e
CFG=${1:-configs/hrsc/redet_l2_finetune.py}
GPU=${GPU:-1}          # 用一张GPU即可（建议 24GB+；也可调小 batch）
STAGE1=8               # 冻结epoch
EPOCHS=36

# 1) Stage-1: 冻结骨干，只训检测头（更快对齐类别边界，召回↑）
python tools/train.py $CFG \
  --work-dir $(python - <<'PY'
import re,sys
p=sys.argv[1]
print(re.sub(r'\.py$','',p).replace('configs/','work_dirs/'))
PY
$CFG) \
  --cfg-options optimizer.lr=0.002 total_epochs=$STAGE1 \
  model.backbone.frozen_stages=4 2>&1 | tee stage1.log

# 2) Stage-2: 解冻全网收敛
python tools/train.py $CFG \
  --resume-from $(ls -t work_dirs/*/latest.pth | head -n1) \
  --cfg-options optimizer.lr=0.001 total_epochs=$EPOCHS model.backbone.frozen_stages=-1 2>&1 | tee stage2.log
