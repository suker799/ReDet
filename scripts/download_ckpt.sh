# scripts/download_ckpt.sh
#!/usr/bin/env bash
set -e
CKPT_URL="https://<your-release-or-original-link>"
OUT="checkpoints/redet_hrsc.pth"
mkdir -p checkpoints
curl -L "$CKPT_URL" -o "$OUT"
# 可选校验
echo "<sha256sum>  $OUT" | sha256sum -c -
