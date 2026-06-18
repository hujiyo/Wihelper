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


def convert(model_path, output_path, opset=ModelConfig.ONNX_OPSET, precision="fp32"):
    """将 PyTorch .pth 模型转换为 ONNX 格式
    precision: "fp32" | "fp16"
    """
    import torch
    from train_model import WiHelperCNN

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA 设备，程序退出")
        sys.exit(1)

    print(f"📦 加载 PyTorch 模型: {model_path}")
    model = WiHelperCNN()
    state_dict = torch.load(model_path, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to("cuda").eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params:,}")

    # FP16 转换：把权重 / 输入 dtype 降下来
    if precision == "fp16":
        model.half()
        print("   精度: FP16（DirectML/CUDA 推理快 2-3 倍）")

    dummy_input = torch.randn(*ModelConfig.INPUT_SHAPE)
    if precision == "fp16":
        dummy_input = dummy_input.half()

    print(f"🔄 导出 ONNX (opset={opset}, precision={precision})...")
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
    precision="fp32",
):
    """验证 ONNX 模型与 PyTorch 模型输出一致性"""
    import torch
    import onnx
    import onnxruntime as ort
    from train_model import WiHelperCNN

    print(f"\n🔍 验证一致性 ({num_samples} 个随机样本)...")

    # PyTorch 模型 - 必须在 GPU 上（与主程序保持一致）
    pt_model = WiHelperCNN()
    state_dict = torch.load(model_path, map_location="cuda", weights_only=True)
    pt_model.load_state_dict(state_dict)
    pt_model = pt_model.to("cuda").eval()
    if precision == "fp16":
        pt_model.half()

    # 自动探测 ONNX 输入 dtype
    onnx_model = onnx.load(onnx_path)
    input_type = onnx_model.graph.input[0].type.tensor_type.elem_type
    onnx_is_fp16 = (input_type == 10)

    # ONNX Runtime 会话 - 必须使用 GPU（DML 优先 → CUDA）
    available_providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in available_providers:
        gpu_providers = ['DmlExecutionProvider']
    elif 'CUDAExecutionProvider' in available_providers:
        gpu_providers = ['CUDAExecutionProvider']
    else:
        print("❌ 未检测到任何 GPU 执行提供者 (DML / CUDA)，无法验证")
        sys.exit(1)
    session = ort.InferenceSession(onnx_path, providers=gpu_providers)

    max_diff = 0.0
    pass_count = 0

    # FP16 验证用更宽松的容忍度（FP16 量化误差在概率空间约 ±0.01）
    eff_tolerance = tolerance if not onnx_is_fp16 else max(tolerance, 0.05)

    np.random.seed(42)
    for i in range(num_samples):
        # 生成随机输入
        img_np = np.random.randint(0, 256, ModelConfig.INPUT_SHAPE).astype(np.float32) / 255.0

        # PyTorch 推理 - PT 模型固定输出 logit (CUDA 上)
        with torch.no_grad():
            pt_input = torch.from_numpy(img_np).to("cuda")
            if precision == "fp16":
                pt_input = pt_input.half()
            pt_logit = float(pt_model(pt_input).cpu().numpy()[0, 0])
            pt_prob = 1.0 / (1.0 + np.exp(-pt_logit))

        # ONNX 推理
        onnx_input = img_np.astype(np.float16) if onnx_is_fp16 else img_np
        onnx_logit = float(
            session.run(None, {ModelConfig.INPUT_NAME: onnx_input})[0][0, 0]
        )
        onnx_prob = 1.0 / (1.0 + np.exp(-onnx_logit))

        # 统一在概率空间比较（避免 logit 空间大动态范围放大 FP16 量化误差）
        diff = abs(pt_prob - onnx_prob)
        max_diff = max(max_diff, diff)

        if diff < eff_tolerance:
            pass_count += 1
        else:
            if i < 5:  # 只打印前5个失败样本
                print(f"   ⚠️ 样本 {i}: PT prob={pt_prob:.6f}, ONNX prob={onnx_prob:.6f}, 差异={diff:.6e}")

        # 进度打印
        if (i + 1) % 20 == 0:
            print(f"   进度: {i + 1}/{num_samples}", flush=True)

    print(f"\n📊 验证结果 (precision={precision}, ONNX={onnx_path}):")
    print(f"   通过: {pass_count}/{num_samples} ({pass_count/num_samples*100:.1f}%)")
    print(f"   最大误差: {max_diff:.6e}")
    print(f"   容忍阈值: {eff_tolerance:.6e}")

    if pass_count == num_samples:
        print("✅ 验证通过！ONNX 模型与 PyTorch 模型输出完全一致")
    elif max_diff < InferenceConfig.VERIFY_ACCEPTABLE_DIFF:
        print("✅ 验证通过！存在微小浮点差异，在实际可接受范围内")
    else:
        print("⚠️ 验证发现显著差异，请检查模型转换")
        sys.exit(1)


def benchmark(onnx_path, num_runs=ModelConfig.BENCHMARK_RUNS_ONNX, warmup_runs=ModelConfig.BENCHMARK_WARMUP_ONNX):
    """基准测试 ONNX Runtime 推理速度"""
    import onnx
    import onnxruntime as ort
    import time

    print(f"\n⚡ ONNX Runtime 推理速度测试 ({num_runs} 次)...")

    # 探测 dtype
    onnx_model = onnx.load(onnx_path)
    input_type = onnx_model.graph.input[0].type.tensor_type.elem_type
    is_fp16 = (input_type == 10)
    np_dtype = np.float16 if is_fp16 else np.float32
    print(f"   精度: {'FP16' if is_fp16 else 'FP32'}")

    # 必须使用 GPU：DML 优先 → CUDA，不允许 CPU 兜底
    available_providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in available_providers:
        providers = ['DmlExecutionProvider']
    elif 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider']
    else:
        print("❌ 未检测到任何 GPU 执行提供者 (DML / CUDA)，无法 benchmark")
        sys.exit(1)
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"   提供者: {providers[0]}")

    # 预热
    dummy = np.random.randint(0, 256, ModelConfig.INPUT_SHAPE).astype(np_dtype) / np.array(255.0, dtype=np_dtype)
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
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16"],
        help="导出精度: fp32(默认) | fp16(GPU 推理更快)",
    )
    parser.add_argument("--skip-verify", action="store_true", help="跳过一致性验证")
    parser.add_argument("--skip-benchmark", action="store_true", help="跳过速度测试")
    parser.add_argument("--num-samples", type=int, default=InferenceConfig.VERIFY_NUM_SAMPLES, help="验证样本数")
    args = parser.parse_args()

    print("=" * 50)
    print("🔄 WiHelper 模型转换工具 (PyTorch → ONNX)")
    print("=" * 50)

    convert(args.model, args.output, args.opset, precision=args.precision)

    if not args.skip_verify:
        verify(args.model, args.output, num_samples=args.num_samples, precision=args.precision)

    if not args.skip_benchmark:
        benchmark(args.output)

    print("\n🎉 转换完成！")
    print(f"   输出文件: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
