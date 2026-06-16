#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练好的 PyTorch 模型转换为 ONNX 格式
用法: python convert_to_onnx.py [--model PATH] [--output PATH]
"""

import os
import sys
import argparse
import numpy as np

from config import (
    AppConfig,
    InferenceConfig,
    ModelConfig,
    PathsConfig,
)


def convert(model_path, output_path, opset=ModelConfig.ONNX_OPSET):
    """将 PyTorch .pth 模型转换为 ONNX 格式"""
    import torch
    from train_model import WiHelperCNN

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    print(f"📦 加载 PyTorch 模型: {model_path}")
    model = WiHelperCNN()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params:,}")

    dummy_input = torch.randn(*ModelConfig.INPUT_SHAPE)

    print(f"🔄 导出 ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=[ModelConfig.INPUT_NAME],
        output_names=[ModelConfig.OUTPUT_NAME],
        dynamic_axes=ModelConfig.DYNAMIC_AXES,  # 固定 batch=1，优化推理性能
        opset_version=opset,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ ONNX 模型已保存: {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def verify(
    model_path,
    onnx_path,
    num_samples=InferenceConfig.VERIFY_NUM_SAMPLES,
    tolerance=InferenceConfig.VERIFY_TOLERANCE,
):
    """验证 ONNX 模型与 PyTorch 模型输出一致性"""
    import torch
    import onnxruntime as ort
    from train_model import WiHelperCNN

    print(f"\n🔍 验证一致性 ({num_samples} 个随机样本)...")

    # PyTorch 模型
    pt_model = WiHelperCNN()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    # ONNX Runtime 会话
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    max_diff = 0.0
    pass_count = 0

    np.random.seed(42)
    for i in range(num_samples):
        # 生成随机输入
        img_np = np.random.randint(0, 256, ModelConfig.INPUT_SHAPE).astype(np.float32) / 255.0

        # PyTorch 推理
        with torch.no_grad():
            pt_input = torch.from_numpy(img_np)
            pt_logit = pt_model(pt_input).numpy()[0, 0]
            pt_prob = 1.0 / (1.0 + np.exp(-pt_logit))

        # ONNX Runtime 推理
        onnx_logit = session.run(None, {ModelConfig.INPUT_NAME: img_np})[0][0, 0]
        onnx_prob = 1.0 / (1.0 + np.exp(-onnx_logit))

        diff = abs(pt_prob - onnx_prob)
        max_diff = max(max_diff, diff)

        if diff < tolerance:
            pass_count += 1
        else:
            if i < 5:  # 只打印前5个失败样本
                print(f"   ⚠️ 样本 {i}: PT={pt_prob:.6f}, ONNX={onnx_prob:.6f}, 差异={diff:.6e}")

    print(f"\n📊 验证结果:")
    print(f"   通过: {pass_count}/{num_samples} ({pass_count/num_samples*100:.1f}%)")
    print(f"   最大误差: {max_diff:.6e}")
    print(f"   容忍阈值: {tolerance:.6e}")

    if pass_count == num_samples:
        print("✅ 验证通过！ONNX 模型与 PyTorch 模型输出完全一致")
    elif max_diff < InferenceConfig.VERIFY_ACCEPTABLE_DIFF:
        print("✅ 验证通过！存在微小浮点差异，在实际可接受范围内")
    else:
        print("⚠️ 验证发现显著差异，请检查模型转换")
        sys.exit(1)


def benchmark(onnx_path, num_runs=ModelConfig.BENCHMARK_RUNS_ONNX, warmup_runs=ModelConfig.BENCHMARK_WARMUP_ONNX):
    """基准测试 ONNX Runtime 推理速度"""
    import onnxruntime as ort
    import time

    print(f"\n⚡ ONNX Runtime 推理速度测试 ({num_runs} 次)...")

    # 尝试 DirectML，不行就用 CPU
    providers = []
    try:
        sess = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider"])
        providers = sess.get_providers()
        print(f"   提供者: {providers[0]}")
    except Exception:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        providers = sess.get_providers()
        print(f"   提供者: {providers[0]}")

    # 预热
    dummy = np.random.randint(0, 256, ModelConfig.INPUT_SHAPE).astype(np.float32) / 255.0
    for _ in range(warmup_runs):
        sess.run(None, {ModelConfig.INPUT_NAME: dummy})

    # 测速
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        sess.run(None, {ModelConfig.INPUT_NAME: dummy})
        times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000
    avg = np.mean(times_ms)
    median = np.median(times_ms)
    p95 = np.percentile(times_ms, 95)
    fps = 1000.0 / avg

    print(f"   平均: {avg:.2f}ms | 中位数: {median:.2f}ms | P95: {p95:.2f}ms | FPS: {fps:.0f}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch → ONNX 模型转换工具")
    parser.add_argument("--model", default=InferenceConfig.DEFAULT_PTH_MODEL_PATH, help="PyTorch 模型路径")
    parser.add_argument("--output", default=AppConfig.DEFAULT_ONNX_MODEL_PATH, help="ONNX 输出路径")
    parser.add_argument("--opset", type=int, default=ModelConfig.ONNX_OPSET, help="ONNX opset 版本")
    parser.add_argument("--skip-verify", action="store_true", help="跳过一致性验证")
    parser.add_argument("--skip-benchmark", action="store_true", help="跳过速度测试")
    parser.add_argument("--num-samples", type=int, default=InferenceConfig.VERIFY_NUM_SAMPLES, help="验证样本数")
    args = parser.parse_args()

    print("=" * 50)
    print("🔄 WiHelper 模型转换工具 (PyTorch → ONNX)")
    print("=" * 50)

    convert(args.model, args.output, args.opset)

    if not args.skip_verify:
        verify(args.model, args.output, num_samples=args.num_samples)

    if not args.skip_benchmark:
        benchmark(args.output)

    print("\n🎉 转换完成！")
    print(f"   输出文件: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
