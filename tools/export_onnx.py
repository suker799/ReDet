#!/usr/bin/env python3
import argparse, torch
from mmrotate.apis import init_model
from mmrotate.apis import inference_detector  # 触发build
from mmrotate.models import build_detector

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=1024)
    args=ap.parse_args()

    model = init_model(args.config, args.ckpt, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    dummy = torch.randn(1,3,args.size,args.size, device=next(model.parameters()).device)
    # 依据你环境的 ops 支持选择 opset（11/12）
    torch.onnx.export(model, dummy, args.out, input_names=['input'], output_names=['boxes','scores','labels'],
                      opset_version=11, do_constant_folding=True, dynamic_axes={'input':{0:'N',2:'H',3:'W'}})
    print('✅ ONNX saved:', args.out)

if __name__=='__main__':
    main()
