#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 多阈值模型评估脚本

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image as PILImage
from train_model import WiHelperCNN


class EvalDataset(datasets.ImageFolder):
    """评估数据集 - 使用 OpenCV 读取 + CenterCrop"""
    def __init__(self, root, transform, crop_size=120):
        super().__init__(root)
        self.transform = transform
        self.crop_size = crop_size

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # CenterCrop 144 → 120
        h, w = img.shape[:2]
        crop_h = (h - self.crop_size) // 2
        crop_w = (w - self.crop_size) // 2
        img = img[crop_h:crop_h + self.crop_size, crop_w:crop_w + self.crop_size]
        img = PILImage.fromarray(img)
        if self.transform:
            img = self.transform(img)
        # 修正标签: ImageFolder gives got=0, nogot=1, we want nogot=0, got=1
        return img, 1 - label


def evaluate_model_at_thresholds(model_path, test_loader, thresholds, device):
    """对单个模型在多个阈值下进行评估"""
    print(f"\n加载模型: {model_path}")
    model = WiHelperCNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    results = []
    for thresh in thresholds:
        y_pred = (y_pred_prob > thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })

    return results

def print_table(model_name, results):
    """打印单个模型的评估表格"""
    print(f"\n{'='*80}")
    print(f" {model_name}")
    print(f"{'='*80}")
    print(f"{'阈值':^8} | {'准确率':^8} | {'精确率':^8} | {'召回率':^8} | {'F1分数':^8} | {'TP':^4} | {'FP':^4} | {'TN':^4} | {'FN':^4}")
    print(f"{'-'*80}")

    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])

    for i, r in enumerate(results):
        marker = " ★" if i == best_f1_idx else ""
        print(f"{r['threshold']:^8.2f} | {r['accuracy']:^8.4f} | {r['precision']:^8.4f} | {r['recall']:^8.4f} | {r['f1']:^8.4f} | {r['tp']:^4d} | {r['fp']:^4d} | {r['tn']:^4d} | {r['fn']:^4d}{marker}")

    print(f"{'-'*80}")
    best = results[best_f1_idx]
    print(f"最佳阈值: {best['threshold']:.2f} (F1={best['f1']:.4f}, 准确率={best['accuracy']:.4f})")

def print_comparison_table(all_results, thresholds):
    """打印所有模型的对比表格"""
    print(f"\n{'='*100}")
    print(f" 各方案最佳阈值对比")
    print(f"{'='*100}")
    print(f"{'方案':^12} | {'最佳阈值':^8} | {'准确率':^8} | {'精确率':^8} | {'召回率':^8} | {'F1分数':^8} | {'TP':^4} | {'FP':^4} | {'TN':^4} | {'FN':^4}")
    print(f"{'-'*100}")

    for model_name, results in all_results.items():
        best_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
        best = results[best_idx]
        print(f"{model_name:^12} | {best['threshold']:^8.2f} | {best['accuracy']:^8.4f} | {best['precision']:^8.4f} | {best['recall']:^8.4f} | {best['f1']:^8.4f} | {best['tp']:^4d} | {best['fp']:^4d} | {best['tn']:^4d} | {best['fn']:^4d}")

    print(f"\n{'='*60}")
    print(f" F1分数排名")
    print(f"{'='*60}")
    sorted_models = sorted(all_results.items(), key=lambda x: max(r['f1'] for r in x[1]), reverse=True)
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        best_f1 = max(r['f1'] for r in results)
        best_thresh = [r['threshold'] for r in results if r['f1'] == best_f1][0]
        print(f"  {rank}. {model_name}: F1={best_f1:.4f} (阈值={best_thresh:.2f})")

def main():
    import sys

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ 使用CPU进行评估")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置
    data_dir = "image"
    batch_size = 32

    # 阈值列表
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    # 模型路径
    model_configs = {
        "方案1": "models-v1.0-1/best_model.pth",
        "方案2": "models-v1.0-2/best_model.pth",
        "方案3": "models-v1.0-3/best_model.pth",
        "方案4": "models-v1.0-4/best_model.pth",
    }

    for name, path in model_configs.items():
        if not os.path.exists(path):
            print(f"警告: {name} 的模型文件不存在: {path}")

    # 创建测试数据集
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = EvalDataset(
        os.path.join(data_dir, 'test'),
        transform=test_transform,
        crop_size=120,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\n测试集样本数: {len(test_dataset)}")
    print(f"类别映射: {test_dataset.class_to_idx}")
    print(f"评估阈值: {thresholds}")

    # 评估所有模型
    all_results = {}
    for model_name, model_path in model_configs.items():
        if os.path.exists(model_path):
            results = evaluate_model_at_thresholds(model_path, test_loader, thresholds, device)
            all_results[model_name] = results
            print_table(model_name, results)

    # 打印对比表格
    if all_results:
        print_comparison_table(all_results, thresholds)

    print("\n评估完成!")

if __name__ == "__main__":
    main()
