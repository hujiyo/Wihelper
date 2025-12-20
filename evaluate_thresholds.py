#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 多阈值模型评估脚本

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import sys

def evaluate_model_at_thresholds(model_path, test_generator, thresholds):
    """对单个模型在多个阈值下进行评估"""
    print(f"\n加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    
    # 获取预测概率
    test_generator.reset()
    y_pred_prob = model.predict(test_generator, verbose=0)
    y_true = test_generator.classes
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_prob > thresh).astype(int).flatten()
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
    
    # 按F1分数排序
    print(f"\n{'='*60}")
    print(f" F1分数排名")
    print(f"{'='*60}")
    sorted_models = sorted(all_results.items(), key=lambda x: max(r['f1'] for r in x[1]), reverse=True)
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        best_f1 = max(r['f1'] for r in results)
        best_thresh = [r['threshold'] for r in results if r['f1'] == best_f1][0]
        print(f"  {rank}. {model_name}: F1={best_f1:.4f} (阈值={best_thresh:.2f})")

def main():
    # 配置
    data_dir = "image"
    img_height, img_width = 144, 144
    batch_size = 32
    
    # 阈值列表
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    # 模型路径
    model_configs = {
        "方案1": "models-v1.0-1/best_model.h5",
        "方案2": "models-v1.0-2/best_model.h5",
        "方案3": "models-v1.0-3/best_model.h5",
        "方案4": "models-v1.0-4/best_model.h5",
    }
    
    # 检查模型文件是否存在
    for name, path in model_configs.items():
        if not os.path.exists(path):
            print(f"警告: {name} 的模型文件不存在: {path}")
    
    # 创建测试数据生成器
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        classes=['nogot', 'got'],
        shuffle=False
    )
    
    print(f"\n测试集样本数: {test_generator.samples}")
    print(f"类别映射: {test_generator.class_indices}")
    print(f"评估阈值: {thresholds}")
    
    # 评估所有模型
    all_results = {}
    for model_name, model_path in model_configs.items():
        if os.path.exists(model_path):
            results = evaluate_model_at_thresholds(model_path, test_generator, thresholds)
            all_results[model_name] = results
            print_table(model_name, results)
    
    # 打印对比表格
    if all_results:
        print_comparison_table(all_results, thresholds)
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()
