#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiHelper - 目标检测推理脚本
加载训练好的模型，进行实时目标检测推理
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import time
from datetime import datetime
import cv2
from train_model import WiHelperCNN


class WiHelperDetector:
    def __init__(self, model_path="models/best_model.pth", threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.img_height = 120
        self.img_width = 120

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                print("请先运行训练脚本训练模型")
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    if models:
                        print("可用模型文件：")
                        print("  - " + "\n  - ".join(models))
                sys.exit(1)

            print(f"📦 加载模型: {self.model_path}")
            self.model = WiHelperCNN()
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ 模型加载成功！")
            print(f"   设备: {self.device}")

            print("\n模型信息:")
            print(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n总参数量: {total_params:,}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请检查：")
            print("  1. 模型文件是否存在")
            print("  2. 模型文件是否损坏")
            sys.exit(1)

    def preprocess_image(self, image, apply_augmentation=False):
        try:
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    if image.shape[-1] == 3:
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    else:
                        image = Image.fromarray(image)
                else:
                    raise ValueError("不支持的图像格式")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_array = np.array(image, dtype=np.float32) / 255.0
            # HWC -> CHW
            img_array = np.transpose(img_array, (2, 0, 1))
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
            return img_tensor

        except Exception as e:
            print(f"❌ 图像预处理失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            return None

    def predict(self, image, return_time=False):
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                if return_time:
                    return False, 0.0, 0.0
                else:
                    return False, 0.0

            start_time = time.time()
            with torch.no_grad():
                output = self.model(processed_image)
                probability = torch.sigmoid(output).item()
            inference_time = time.time() - start_time

            has_target = probability >= self.threshold

            if return_time:
                return has_target, probability, inference_time
            else:
                return has_target, probability

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            if return_time:
                return False, 0.0, 0.0
            else:
                return False, 0.0

    def predict_from_file(self, image_path, show_details=True, return_time=False):
        try:
            if not os.path.exists(image_path):
                print(f"❌ 文件不存在: {image_path}")
                return (False, 0.0, 0.0) if return_time else (False, 0.0)

            if not self._is_image_file(image_path):
                print(f"❌ 文件不是支持的图片格式: {image_path}")
                return (False, 0.0, 0.0) if return_time else (False, 0.0)

            image = Image.open(image_path)
            has_target, probability, inference_time = self.predict(image, return_time=True)

            if show_details:
                result = "有目标" if has_target else "无目标"
                status_icon = "🎯" if has_target else "❌"
                confidence_level = self._get_confidence_level(probability)

                print(f"{status_icon} 预测结果: {result}")
                print(f"   概率: {probability:.4f} ({confidence_level})")
                print(f"   阈值: {self.threshold:.2f}")
                print(f"   文件: {os.path.basename(image_path)}")
                print(f"   推理时间: {inference_time*1000:.1f}ms")

                if probability >= self.threshold:
                    print("   状态: ✓ 超过阈值")
                else:
                    print("   状态: ✗ 未达到阈值")

            if return_time:
                return has_target, probability, inference_time
            else:
                return has_target, probability

        except Exception as e:
            print(f"❌ 文件预测失败: {e}")
            print(f"   文件: {os.path.basename(image_path) if 'image_path' in locals() else 'Unknown'}")
            print(f"   错误类型: {type(e).__name__}")
            return (False, 0.0, 0.0) if return_time else (False, 0.0)

    def _is_image_file(self, file_path):
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        _, ext = os.path.splitext(file_path.lower())
        return ext in supported_extensions

    def _get_confidence_level(self, probability):
        if probability >= 0.9:
            return "非常高"
        elif probability >= 0.8:
            return "高"
        elif probability >= 0.7:
            return "中等"
        elif probability >= 0.6:
            return "较低"
        else:
            return "很低"

    def batch_predict(self, image_dir, save_results=True):
        if not os.path.exists(image_dir):
            print(f"❌ 目录不存在: {image_dir}")
            return []

        results = []
        image_files = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if self._is_image_file(file):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            print(f"❌ 在目录 {image_dir} 中未找到支持的图片文件")
            return []

        print(f"📁 开始批量预测，共 {len(image_files)} 张图片")
        print(f"   目录: {os.path.abspath(image_dir)}")
        print(f"   阈值: {self.threshold:.2f}")

        print("   正在预加载图片到内存...")
        loaded_images = []
        valid_image_files = []

        for image_path in image_files:
            try:
                image = Image.open(image_path)
                loaded_images.append(image)
                valid_image_files.append(image_path)
            except Exception as e:
                print(f"   ⚠️  无法加载图片 {os.path.basename(image_path)}: {e}")
                continue

        print(f"   ✓ 成功加载 {len(loaded_images)} 张图片")

        stats = {
            'total': len(loaded_images),
            'processed': 0,
            'correct': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_time': 0.0
        }

        start_time = time.time()

        for i, (image_path, image) in enumerate(zip(valid_image_files, loaded_images)):
            has_target, probability, inference_time = self.predict(image, return_time=True)
            stats['total_time'] += inference_time

            true_label = self._infer_true_label_from_filename(image_path)
            pred_label = 1 if has_target else 0

            is_correct = true_label == pred_label
            if is_correct:
                stats['correct'] += 1

            if true_label == 1 and pred_label == 1:
                stats['true_positives'] += 1
            elif true_label == 0 and pred_label == 1:
                stats['false_positives'] += 1
            elif true_label == 1 and pred_label == 0:
                stats['false_negatives'] += 1
            elif true_label == 0 and pred_label == 0:
                stats['true_negatives'] += 1

            stats['processed'] += 1

            result = {
                'file': os.path.basename(image_path),
                'path': image_path,
                'prediction': '有目标' if has_target else '无目标',
                'probability': round(probability, 4),
                'true_label': '有目标' if true_label else '无目标',
                'correct': is_correct,
                'inference_time_ms': round(inference_time * 1000, 2)
            }
            results.append(result)

        total_time = time.time() - start_time

        self._print_batch_results(stats, total_time)

        if save_results:
            results_file = self._save_results_to_file(results, stats, total_time)

        return results

    def _infer_true_label_from_filename(self, image_path):
        filename = os.path.basename(image_path).lower()
        target_keywords = ['target', 'got', 'hit', 'aim']
        nogot_keywords = ['nogot', 'miss', 'nohit', 'notarget']

        for keyword in target_keywords:
            if keyword in filename:
                return 1
        for keyword in nogot_keywords:
            if keyword in filename:
                return 0

        path_parts = image_path.lower().split(os.sep)
        if 'got' in path_parts:
            return 1
        elif 'nogot' in path_parts:
            return 0
        return 0

    def _print_batch_results(self, stats, total_time):
        print("\n" + "="*60)
        print("📊 批量预测结果统计")
        print("="*60)

        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (stats['true_positives'] + stats['false_positives']) > 0 else 0
        recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives']) if (stats['true_positives'] + stats['false_negatives']) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"总体统计:")
        print(f"  - 总图片数: {stats['total']}")
        print(f"  - 处理图片数: {stats['processed']}")
        print(f"  - 准确率: {accuracy:.4f}")
        print(f"  - 精确率: {precision:.4f}")
        print(f"  - 召回率: {recall:.4f}")
        print(f"  - F1分数: {f1_score:.4f}")

        print(f"\n混淆矩阵:")
        print(f"  - 真正例 (TP): {stats['true_positives']} (正确预测有目标)")
        print(f"  - 假正例 (FP): {stats['false_positives']} (错误预测有目标)")
        print(f"  - 真负例 (TN): {stats['true_negatives']} (正确预测无目标)")
        print(f"  - 假负例 (FN): {stats['false_negatives']} (错误预测无目标)")

        print(f"\n性能统计:")
        print(f"  - 总耗时: {total_time:.2f}秒")
        print(f"  - 平均推理时间: {stats['total_time']/stats['processed']*1000:.2f}ms/张")
        print(f"  - 处理速度: {stats['processed']/total_time:.1f}张/秒")

    def _save_results_to_file(self, results, stats, total_time):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"batch_prediction_results_{timestamp}.json"

        import json

        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

        save_data = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'threshold': self.threshold,
            'total_images': stats['total'],
            'processed_images': stats['processed'],
            'accuracy': round(accuracy, 4),
            'total_time_seconds': round(total_time, 2),
            'average_inference_time_ms': round(stats['total_time'] / stats['processed'] * 1000, 2),
            'statistics': {
                'true_positives': stats['true_positives'],
                'false_positives': stats['false_positives'],
                'true_negatives': stats['true_negatives'],
                'false_negatives': stats['false_negatives']
            },
            'results': results
        }

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"📄 详细结果已保存到: {results_file}")
            return results_file
        except Exception as e:
            print(f"❌ 保存结果文件失败: {e}")
            return None

    def benchmark_inference_speed(self, num_runs=100, warmup_runs=10):
        print(f"\n⚡ 测试推理速度...")
        print(f"   预热轮次: {warmup_runs}")
        print(f"   测试轮次: {num_runs}")
        print(f"   图像尺寸: {self.img_height}x{self.img_width}")
        print(f"   设备: {self.device}")

        test_images = []
        for i in range(min(5, num_runs)):
            img_array = np.random.randint(0, 255, (self.img_height, self.img_width, 3), dtype=np.uint8)
            test_images.append(Image.fromarray(img_array))

        print("   正在预热...")
        for i in range(warmup_runs):
            img = test_images[i % len(test_images)]
            self.predict(img, return_time=False)
        print("   ✓ 预热完成")

        print("   正在测试...")
        times = []

        for i in range(num_runs):
            img = test_images[i % len(test_images)]
            start_time = time.time()
            has_target, probability = self.predict(img, return_time=False)
            end_time = time.time()
            times.append(end_time - start_time)

        times_ms = np.array(times) * 1000
        avg_time = np.mean(times_ms)
        min_time = np.min(times_ms)
        max_time = np.max(times_ms)
        std_time = np.std(times_ms)
        median_time = np.median(times_ms)
        p95_time = np.percentile(times_ms, 95)
        p99_time = np.percentile(times_ms, 99)
        avg_fps = 1000 / avg_time if avg_time > 0 else 0

        print("\n📊 推理速度统计")
        print("="*50)
        print(f"平均推理时间: {avg_time:.2f}ms")
        print(f"最小推理时间: {min_time:.2f}ms")
        print(f"最大推理时间: {max_time:.2f}ms")
        print(f"标准差: {std_time:.2f}ms")
        print(f"中位数: {median_time:.2f}ms")
        print(f"P95: {p95_time:.2f}ms")
        print(f"P99: {p99_time:.2f}ms")
        print(f"平均FPS: {avg_fps:.2f}")

        print("\n性能评估:")
        if avg_fps >= 100:
            print("   性能等级: ⭐⭐⭐⭐⭐ 极高")
        elif avg_fps >= 50:
            print("   性能等级: ⭐⭐⭐⭐ 高")
        elif avg_fps >= 20:
            print("   性能等级: ⭐⭐⭐ 中等")
        elif avg_fps >= 10:
            print("   性能等级: ⭐⭐ 较低")
        else:
            print("   性能等级: ⭐ 很低")

        if avg_time > 50:
            print("   💡 建议: 推理时间较长，考虑优化模型或使用GPU")
        elif avg_time > 20:
            print("   💡 建议: 推理时间适中，可接受")
        else:
            print("   ✅ 推理性能优秀！")

        return {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'median_time_ms': median_time,
            'p95_time_ms': p95_time,
            'p99_time_ms': p99_time,
            'fps': avg_fps,
            'num_runs': num_runs
        }

def main():
    import sys

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ 未检测到GPU设备，使用CPU推理")

    import argparse

    parser = argparse.ArgumentParser(
        description='WiHelper - 目标检测推理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python inference.py --image test.png
  python inference.py --batch image/test/
  python inference.py --benchmark
  python inference.py --model models/final_model.pth --threshold 0.7 --batch image/test/
        """
    )

    parser.add_argument('--model', default='models/best_model.pth',
                       help='模型文件路径 (默认: models/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='预测阈值 (0-1, 默认: 0.8)')
    parser.add_argument('--image', help='单张图片文件路径')
    parser.add_argument('--batch', help='批量预测目录路径')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行推理速度测试')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式 (减少输出信息)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存批量预测结果到文件')

    args = parser.parse_args()

    print("="*60)
    print("🎯 WiHelper - 目标检测推理工具")
    print("="*60)
    print(f"模型路径: {args.model}")
    print(f"预测阈值: {args.threshold}")
    print(f"静默模式: {'是' if args.quiet else '否'}")

    try:
        detector = WiHelperDetector(args.model, args.threshold)

        if args.image:
            print(f"\n🔍 预测单张图片: {args.image}")
            detector.predict_from_file(args.image, show_details=not args.quiet)

        elif args.batch:
            print(f"\n📁 批量预测目录: {args.batch}")
            detector.batch_predict(args.batch, save_results=not args.no_save)

        elif args.benchmark:
            benchmark_results = detector.benchmark_inference_speed()
            if args.quiet:
                print(f"平均FPS: {benchmark_results['fps']:.1f}")

        else:
            print("\n请指定操作：")
            print("  --image <图片路径>    : 预测单张图片")
            print("  --batch <目录路径>    : 批量预测目录")
            print("  --benchmark           : 运行推理速度测试")
            print("\n可选参数：")
            print("  --model <路径>        : 指定模型文件")
            print("  --threshold <值>      : 设置预测阈值")
            print("  --quiet               : 静默模式")
            print("  --no-save             : 不保存批量预测结果")
            print("\n完整示例：")
            print("  python inference.py --model models/final_model.pth --threshold 0.7 --image test.png")

    except KeyboardInterrupt:
        print("\n👋 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
