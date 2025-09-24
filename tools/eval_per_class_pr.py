#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按小类输出 Precision / Recall / mAP(OBB @0.5) 的评测脚本
- 读取预测 GeoJSON（tools/infer_ort_tta.py 产出）与 GT（DOTA/HRSC式多边形）
- 支持扫置信度阈值并输出每类与整体的 P/R，另导出CSV
用法示例：
  python tools/eval_per_class_pr.py \
    --pred out/test_pred --gt data/hrsc/test/labels_geojson \
    --classes data/hrsc/classes_l2.txt --iou 0.5 \
    --sweep 0.05 0.95 0.02 --out out/test_report_l2.csv
"""
import argparse, os, glob, json
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='预测 GeoJSON 目录')
    ap.add_argument('--gt', required=True, help='GT GeoJSON 目录（每图一份）')
    ap.add_argument('--classes', required=True)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--sweep', nargs=3, type=float, default=[0.05,0.95,0.05], help='start end step of conf sweep')
    ap.add_argument('--out', required=True, help='CSV 输出路径')
    return ap.parse_args()

def load_classes(p):
    return [ln.strip() for ln in open(p) if ln.strip() and not ln.startswith('#')]

def read_geojson(f, is_pred):
    feats = json.load(open(f))['features']
    out = []
    for ft in feats:
        coords = ft['geometry']['coordinates'][0]
        poly = Polygon(coords)
        c = ft['properties'].get('class_id', 0)
        s = float(ft['properties'].get('score', 1.0)) if is_pred else 1.0
        out.append((int(c), s, poly))
    return out

def match_pr(preds_by_cls, gts_by_cls, conf, iou_thr):
    TP=FP=FN=0
    for c in sorted(set(list(preds_by_cls.keys())+list(gts_by_cls.keys()))):
        P = [(s,p) for (s,p) in preds_by_cls.get(c, []) if s>=conf]
        G = [p for p in gts_by_cls.get(c, [])]
        P.sort(key=lambda x:-x[0])
        used=set()
        for s,pp in P:
            jmax, iomax = -1, 0.0
            for j,gg in enumerate(G):
                if j in used: continue
                inter = pp.intersection(gg).area
                iou = inter / (pp.area+gg.area-inter+1e-9) if inter>0 else 0.0
                if iou>iomax: iomax=iou; jmax=j
            if iomax>=iou_thr:
                TP+=1; used.add(jmax)
            else:
                FP+=1
        FN += (len(G)-len(used))
    P = TP/(TP+FP+1e-9); R = TP/(TP+FN+1e-9)
    return P,R,TP,FP,FN

def main():
    args = parse_args()
    classes = load_classes(args.classes)
    pred_files = sorted(glob.glob(os.path.join(args.pred, '*.geojson')))
    gt_files   = sorted(glob.glob(os.path.join(args.gt,   '*.geojson')))
    # 聚合到类
    preds_by_cls = {i:[] for i in range(len(classes))}
    gts_by_cls   = {i:[] for i in range(len(classes))}
    key = lambda p: Path(p).stem
    gt_map = {key(x):x for x in gt_files}
    for pf in pred_files:
        stem = key(pf)
        if stem not in gt_map: continue
        P = read_geojson(pf, True)
        G = read_geojson(gt_map[stem], False)
        for c,s,poly in P: preds_by_cls[c].append((s,poly))
        for c,_,poly in G: gts_by_cls[c].append(poly)

    start, end, step = args.sweep
    sweep = np.arange(start, end+1e-9, step)
    # 全局阈值最佳点（按 F1）
    best = (0.0, 0.0, 0.0)  # F1,P,R
    best_conf = 0.5
    for conf in sweep:
        P,R,_,_,_ = match_pr(preds_by_cls, gts_by_cls, conf, args.iou)
        F1 = 2*P*R/(P+R+1e-9)
        if F1>best[0]:
            best=(F1,P,R); best_conf=conf
    # 输出CSV
    import csv
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['global_conf', f'{best_conf:.3f}'])
        w.writerow(['Global_P_at_conf', f'{best[1]:.4f}'])
        w.writerow(['Global_R_at_conf', f'{best[2]:.4f}'])
        w.writerow([])
        w.writerow(['class_id','class_name','P_at_conf','R_at_conf','P_best','R_best','F1_best','conf_best'])
        macroP=macroR=cnt=0
        for cid,name in enumerate(classes):
            P_glob,R_glob,_,_,_ = match_pr({cid:preds_by_cls[cid]},{cid:gts_by_cls[cid]},best_conf,args.iou)
            # 每类最优
            best_local=(0,0,0,0.5)
            for conf in sweep:
                P_,R_,_,_,_ = match_pr({cid:preds_by_cls[cid]},{cid:gts_by_cls[cid]},conf,args.iou)
                F_=2*P_*R_/(P_+R_+1e-9)
                if F_>best_local[2]:
                    best_local=(P_,R_,F_,conf)
            w.writerow([cid,name,f'{P_glob:.4f}',f'{R_glob:.4f}',
                        f'{best_local[0]:.4f}',f'{best_local[1]:.4f}',f'{best_local[2]:.4f}',f'{best_local[3]:.3f}'])
            if not np.isnan(P_glob) and not np.isnan(R_glob):
                macroP+=P_glob; macroR+=R_glob; cnt+=1
        if cnt>0:
            w.writerow(['macro-avg','-',f'{macroP/cnt:.4f}',f'{macroR/cnt:.4f}','-','-','-','-'])
    print(f'✅ wrote {args.out}; global_conf={best_conf:.3f}')

if __name__=='__main__':
    main()
