#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Runtime（CPU）推理脚本（支持单通道→伪RGB、滑窗、TTA、soft-NMS、旋转框输出为 GeoJSON）
用法示例：
  python tools/infer_ort_tta.py \
    --onnx out/redet_l2.onnx --images data/hrsc/test/images \
    --imgsz 1024 --overlap 0.5 \
    --conf 0.15 --soft-nms --tta-scales 0.75,1.0,1.25 --tta-angles -10,0,10 \
    --gray-to-rgb --classes data/hrsc/classes_l2.txt \
    --out out/test_pred
"""
import argparse, os, json, math, glob
from pathlib import Path
import numpy as np
import onnxruntime as ort
import cv2
from shapely.geometry import Polygon, mapping

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--images', required=True, help='目录，包含影像（.tif/.png/.jpg）')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--overlap', type=float, default=0.5, help='滑窗重叠比例 [0,1)')
    ap.add_argument('--conf', type=float, default=0.2, help='置信度阈值（全局）')
    ap.add_argument('--nms-iou', type=float, default=0.6)
    ap.add_argument('--soft-nms', action='store_true')
    ap.add_argument('--tta-scales', type=str, default='', help='逗号分隔，如 0.75,1.0,1.25')
    ap.add_argument('--tta-angles', type=str, default='', help='逗号分隔，如 -10,0,10（度）')
    ap.add_argument('--gray-to-rgb', action='store_true', help='单通道→伪RGB（三通道复制）')
    ap.add_argument('--classes', type=str, default='', help='可选：类别文件（逐行一个类别名）')
    ap.add_argument('--out', required=True)
    return ap.parse_args()

def load_classes(path):
    if not path or not os.path.isfile(path): return None
    with open(path, 'r') as f:
        names = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    return names

def img_read(path, gray_to_rgb=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:  # single channel
        if gray_to_rgb:
            img = np.stack([img, img, img], axis=-1)
        else:
            # 如果模型是1通道输入，这里可保持单通道；大多数ReDet导出为3通道，故默认伪RGB
            img = np.stack([img, img, img], axis=-1)
    elif img.shape[2]==4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def preprocess(img, size):
    h, w = img.shape[:2]
    if (h, w) != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = x.transpose(2,0,1)[None, :, :, :]  # 1x3xHxW
    return x

def obb_to_poly(cx, cy, w, h, angle_deg):
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx = np.array([[ w/2,  h/2],
                   [ w/2, -h/2],
                   [-w/2, -h/2],
                   [-w/2,  h/2]], dtype=np.float32)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    pts = dx @ R.T + np.array([cx, cy], dtype=np.float32)
    return pts

def nms_rotated(boxes, scores, iou_thr=0.6, soft=False, sigma=0.5):
    """简单的 rotated NMS：boxes = [N,5] (cx,cy,w,h,angle_deg)"""
    keep = []
    idxs = scores.argsort()[::-1]
    polys = None
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        if polys is None:
            polys = [Polygon(obb_to_poly(*b)) for b in boxes]
        iou = []
        for j in idxs[1:]:
            inter = polys[i].intersection(polys[j]).area
            if inter <= 0:
                iou.append(0.0)
            else:
                iou.append(inter / (polys[i].area + polys[j].area - inter + 1e-9))
        iou = np.array(iou, dtype=np.float32)
        rest = idxs[1:]
        if not soft:
            rest = rest[iou < iou_thr]
        else:
            # Soft-NMS 高斯
            scores[rest] = scores[rest] * np.exp(-(iou**2)/sigma)
            rest = rest[scores[rest] >= 1e-6]
        idxs = rest
        if soft:
            idxs = idxs[np.argsort(scores[idxs])[::-1]]
    return keep

def slide_windows(h, w, size, overlap):
    step = int(size * (1.0 - overlap))
    xs = list(range(0, max(1, w-size+1), step))
    ys = list(range(0, max(1, h-size+1), step))
    if len(xs)==0: xs=[0]
    if len(ys)==0: ys=[0]
    return [(x,y) for y in ys for x in xs]

def run_session(sess, x):
    # 适配常见导出：输入名取第一个
    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: x})
    # 需根据导出的 ReDet onnx 输出格式适配：
    # 假设 outputs: boxes [N,5(cx,cy,w,h,angle_deg)], scores [N], labels [N]
    # 如果你的导出格式不同，请在此解析调整：
    if len(outs)==3:
        boxes, scores, labels = outs
    else:
        # 兜底：尝试按常见顺序
        boxes, scores, labels = outs[0], outs[1], outs[2]
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int32)

def tta_transforms(scales, angles):
    scs = [float(s) for s in scales.split(',') if s] if scales else [1.0]
    angs = [float(a) for a in angles.split(',') if a] if angles else [0.0]
    return scs, angs

def main():
    args = parse_args()
    names = load_classes(args.classes)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    img_paths = sorted(sum([glob.glob(os.path.join(args.images, ext)) for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff')], []))
    scales, angles = tta_transforms(args.tta_scales, args.tta_angles)

    for p in img_paths:
        im0 = img_read(p, gray_to_rgb=args.gray_to_rgb)
        H0, W0 = im0.shape[:2]
        tiles = slide_windows(H0, W0, args.imgsz, args.overlap)
        all_boxes, all_scores, all_labels = [], [], []
        for (x0,y0) in tiles:
            chip = im0[y0:y0+args.imgsz, x0:x0+args.imgsz]
            if chip.shape[0] != args.imgsz or chip.shape[1] != args.imgsz:
                pad = np.zeros((args.imgsz, args.imgsz, 3), dtype=chip.dtype)
                pad[:chip.shape[0], :chip.shape[1]] = chip
                chip = pad
            for s in scales:
                chip_s = cv2.resize(chip, (int(args.imgsz*s), int(args.imgsz*s)), interpolation=cv2.INTER_LINEAR)
                for a in angles:
                    M = cv2.getRotationMatrix2D((chip_s.shape[1]/2, chip_s.shape[0]/2), a, 1.0)
                    chip_sa = cv2.warpAffine(chip_s, M, (chip_s.shape[1], chip_s.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                    xin = preprocess(chip_sa, args.imgsz)
                    boxes, scores, labels = run_session(sess, xin)
                    # 反变换回 chip 坐标系
                    # 假设 boxes 是 (cx,cy,w,h,angle_deg) 在网络输入坐标
                    scale_back = args.imgsz / max(1, chip_s.shape[0])  # 近似等比
                    boxes[:, :4] /= scale_back
                    boxes[:, 0] -= 0  # cx
                    boxes[:, 1] -= 0  # cy
                    boxes[:, 4] += (-a)  # 角度回退
                    # 平移回原图
                    boxes[:, 0] += x0
                    boxes[:, 1] += y0
                    # 过滤低分
                    m = scores >= args.conf
                    all_boxes.append(boxes[m])
                    all_scores.append(scores[m])
                    all_labels.append(labels[m])
        if len(all_boxes):
            boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0,5),np.float32)
            scores = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,),np.float32)
            labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,),np.int32)
        else:
            boxes = np.zeros((0,5),np.float32); scores = np.zeros((0,),np.float32); labels = np.zeros((0,),np.int32)
        # NMS（按类分别执行）
        feats=[]
        for c in np.unique(labels):
            idx = np.where(labels==c)[0]
            if idx.size==0: continue
            keep = nms_rotated(boxes[idx], scores[idx], iou_thr=args.nms_iou, soft=args.soft_nms, sigma=0.5)
            for k in keep:
                cx,cy,w,h,ang = boxes[idx][k].tolist()
                poly = obb_to_poly(cx,cy,w,h,ang).tolist()
                props = {"class_id": int(c), "score": float(scores[idx][k])}
                if names: props["class_name"] = names[int(c)] if int(c) < len(names) else str(c)
                feats.append({"type":"Feature",
                              "geometry":{"type":"Polygon","coordinates":[poly+[poly[0]]]},
                              "properties": props})
        out_geo = Path(args.out)/f"{Path(p).stem}.geojson"
        json.dump({"type":"FeatureCollection","features":feats}, open(out_geo,"w"))
        print(f"[OK] {p} -> {out_geo} (n={len(feats)})")

if __name__ == "__main__":
    main()
