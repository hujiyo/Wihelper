#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查找训练集中模型预测与人工标注不一致的样本
将不一致的图片复制到 image/ 下，供 select_helper 重新标注
"""

import os
import sys
import shutil
import numpy as np
import torch
from PIL import Image
from train_model import WiHelperCNN, preprocess, find_best_model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="查找训练集中模型预测与人工标注不一致的样本")
    parser.add_argument("--uncertain", action="store_true",
                        help="同时提取预测概率在 0.2~0.8 之间的不确定样本")
    args = parser.parse_args()

    model_path = find_best_model()
    if model_path is None:
        print("未找到 best_model.pth，请确认 models/ 目录下有模型文件")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WiHelperCNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"设备: {device}")
    if args.uncertain:
        print("模式: 不一致 + 不确定(0.2~0.8)")

    threshold = 0.5
    out_dir = "image"
    os.makedirs(out_dir, exist_ok=True)

    folders = {
        "image/train/got": 1,      # 人工标注: 有目标
        "image/train/nogot": 0,     # 人工标注: 无目标
    }

    total = 0
    mismatch = 0
    uncertain_count = 0
    mismatch_list = []

    for folder, true_label in folders.items():
        if not os.path.isdir(folder):
            print(f"跳过不存在的目录: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        label_name = "有目标(got)" if true_label == 1 else "无目标(nogot)"
        print(f"\n扫描 {folder}/ ({len(files)} 张, 标签={label_name})")

        for fname in files:
            fpath = os.path.join(folder, fname)
            tensor = preprocess(fpath)
            if tensor is None:
                print(f"  跳过无法读取的文件: {fname}")
                continue

            total += 1
            with torch.no_grad():
                logit = model(tensor.to(device))
                prob = torch.sigmoid(logit).item()

            pred_label = 1 if prob >= threshold else 0

            is_mismatch = pred_label != true_label
            is_uncertain = args.uncertain and 0.2 <= prob <= 0.8

            if is_mismatch:
                mismatch += 1
                mismatch_list.append((fpath, fname, true_label, pred_label, prob, "标签不一致"))
            elif is_uncertain:
                uncertain_count += 1
                mismatch_list.append((fpath, fname, true_label, pred_label, prob, "不确定"))

            if is_mismatch or is_uncertain:
                dst = os.path.join(out_dir, fname)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(fname)
                    dst = os.path.join(out_dir, f"{base}_relabel{ext}")
                shutil.move(fpath, dst)

    flagged = mismatch + uncertain_count
    print(f"\n{'='*50}")
    print(f"扫描完成: 共 {total} 张, 不一致 {mismatch} 张 ({mismatch/total*100:.1f}%)")
    if args.uncertain:
        print(f"不确定(0.2~0.8): {uncertain_count} 张 ({uncertain_count/total*100:.1f}%)")
    print(f"合计标记: {flagged} 张 ({flagged/total*100:.1f}%)")
    print(f"{'='*50}")

    if mismatch_list:
        print(f"\n标记样本详情 (已移动到 {out_dir}/):")
        print(f"{'文件名':<50} {'人工标注':<8} {'模型预测':<8} {'概率':<8} {'原因':<10}")
        print("-" * 88)
        for fpath, fname, true_l, pred_l, prob, reason in sorted(mismatch_list, key=lambda x: x[4], reverse=True):
            true_name = "有目标" if true_l == 1 else "无目标"
            pred_name = "有目标" if pred_l == 1 else "无目标"
            print(f"{fname:<50} {true_name:<8} {pred_name:<8} {prob:.4f}  {reason}")

        print(f"\n请运行 select_helper.py 重新标注 image/ 下的 {flagged} 张图片")
        print("标记的图片已从 train/ 移动到 image/，标注完成后无需手动删除")
    else:
        print("所有样本的模型预测与人工标注一致!")


if __name__ == "__main__":
    main()
