#!/usr/bin/env bash
set -e

# 使用文件ID
FILE_ID="1vTU6OeFD6CX4zkQn7szlgL7Qc_MOZpgC"
OUT="checkpoints/re_resnet50_c8_batch256-12933bc2.pth"

mkdir -p checkpoints

# 安装gdown（如果尚未安装）
pip install gdown

# 使用gdown下载
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUT"
