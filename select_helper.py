# -*- coding: utf-8 -*-
"""
数据标注助手
功能：
- 逐个显示图片
- 空格键：无目标 (nogot)
- 回车键：有目标 (got)
- Delete键：删除图片（质量不高）
- Ctrl+Z：回退上一次操作 (最多5步，包括删除操作)
- 自动移动文件到训练集对应文件夹
- 启动时预推理所有图片，显示模型建议和置信度
- 简单直接：有文件就处理，没文件就结束
"""

import os, sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import shutil
import numpy as np
import torch
import torch.nn as nn
import cv2


class WiHelperCNN(nn.Module):
    """空洞卷积版 - 全程保持空间分辨率，中心裁剪后压缩"""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.compress = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        h = (x.shape[2] - 96) // 2
        w = (x.shape[3] - 96) // 2
        x = x[:, :, h:h+96, w:w+96]
        x = self.compress(x)
        x = self.classifier(x)
        return x


def center_crop(img, size=120):
    h, w = img.shape[:2]
    ch = (h - size) // 2
    cw = (w - size) // 2
    return img[ch:ch + size, cw:cw + size]


def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = center_crop(img, 120)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def find_best_model():
    candidates = ["models/best_model.pth", "models-v1.1-4/best_model.pth"]
    for p in candidates:
        if os.path.exists(p):
            return p
    for d in ["models", "models-v1.1-4"]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "best" in f.lower() and f.endswith(".pth"):
                    return os.path.join(d, f)
    return None


class SelectHelper:
    def __init__(self, source_dir="image", target_base="image"):
        self.source_dir = source_dir
        self.target_base = target_base

        self.train_target_dir = os.path.join(target_base, "train", "got")
        self.train_notarget_dir = os.path.join(target_base, "train", "nogot")

        for dir_path in [self.train_target_dir, self.train_notarget_dir]:
            if os.path.exists(dir_path):
                existing_files = os.listdir(dir_path)
                if existing_files:
                    print(f"  文件夹已存在且包含 {len(existing_files)} 个文件: {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  创建新文件夹: {dir_path}")

        self.image_files = []
        self.current_index = 0

        self.stats = {
            "train_target": 0,
            "train_notarget": 0,
            "deleted": 0,
            "total_processed": 0
        }

        self.undo_stack = []
        self.max_undo = 5

        self.root = None
        self.image_label = None
        self.info_label = None
        self.prediction_label = None
        self.current_image = None

        # 模型预测结果: {filepath: (prob, pred_label)}
        self.predictions = {}
        self.model = None
        self.device = None

    def load_model_and_predict(self, image_files):
        """启动时加载模型并批量推理所有图片"""
        model_path = find_best_model()
        if model_path is None:
            print("  未找到模型文件，跳过模型预测")
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = WiHelperCNN()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"  模型: {os.path.basename(model_path)} ({self.device})")
        print(f"  预推理 {len(image_files)} 张图片...")

        for i, fpath in enumerate(image_files):
            tensor = preprocess(fpath)
            if tensor is None:
                continue
            with torch.no_grad():
                logit = self.model(tensor.to(self.device))
                prob = torch.sigmoid(logit).item()
            pred_label = 1 if prob >= 0.5 else 0
            self.predictions[fpath] = (prob, pred_label)

            if (i + 1) % 50 == 0 or i + 1 == len(image_files):
                print(f"    {i + 1}/{len(image_files)}")

        print(f"  预推理完成")

    def get_image_files(self):
        if not os.path.exists(self.source_dir):
            print(f"  源文件夹不存在: {self.source_dir}")
            return []

        all_files = []
        try:
            for file in os.listdir(self.source_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(self.source_dir, file)
                    if os.path.isfile(full_path):
                        all_files.append(full_path)
        except Exception as e:
            print(f"  读取目录失败: {e}")
            return []

        all_files.sort()
        print(f"  找到 {len(all_files)} 张图片文件")
        return all_files

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("数据标注助手")
        self.root.geometry("520x700")
        self.root.resizable(True, True)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="数据标注助手", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 5))

        # 模型预测显示区域 - 显眼的大字
        self.prediction_label = tk.Label(
            main_frame, text="", font=("Arial", 18, "bold"),
            fg="white", bg="gray", pady=8
        )
        self.prediction_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # 图片显示区域
        self.image_label = ttk.Label(main_frame, text="加载中...", background="lightgray")
        self.image_label.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))

        # 信息显示
        self.info_label = ttk.Label(main_frame, text="准备开始...", font=("Arial", 10))
        self.info_label.grid(row=3, column=0, pady=(0, 5))

        instructions = """
操作说明：
  空格键：没中 (nogot)
  回车键：中了 (got)
  Delete键：删除图片
  Ctrl+Z：回退 (最多5步)
  Ctrl+C：退出
        """
        instr_label = ttk.Label(main_frame, text=instructions, font=("Arial", 9), justify=tk.LEFT)
        instr_label.grid(row=4, column=0, pady=(0, 5))

        self.root.bind('<space>', lambda e: self.annotate_image(False))
        self.root.bind('<Return>', lambda e: self.annotate_image(True))
        self.root.bind('<Delete>', lambda e: self.delete_image())
        self.root.bind('<Control-z>', lambda e: self.undo_last_annotation())

        self.root.protocol("WM_DELETE_WINDOW", self.quit_program)

    def update_prediction_display(self):
        """更新模型预测显示"""
        if self.current_index >= len(self.image_files):
            self.prediction_label.config(text="", bg="gray")
            return

        current_file = self.image_files[self.current_index]
        if current_file not in self.predictions:
            self.prediction_label.config(text="无模型预测", bg="gray", fg="white")
            return

        prob, pred_label = self.predictions[current_file]

        if pred_label == 1:
            # 模型认为有目标
            if prob >= 0.8:
                bg_color = "#CC0000"  # 深红 - 高置信度有目标
                text = f"模型建议: 中了  ({prob:.1%})"
            elif prob >= 0.6:
                bg_color = "#DD6644"  # 橙红 - 中等置信度
                text = f"模型建议: 可能中了  ({prob:.1%})"
            else:
                bg_color = "#DDAA44"  # 橙黄 - 低置信度
                text = f"模型建议: 倾向中了  ({prob:.1%})"
            fg_color = "white"
        else:
            # 模型认为无目标
            if prob <= 0.2:
                bg_color = "#0066CC"  # 深蓝 - 高置信度无目标
                text = f"模型建议: 没中  ({prob:.1%})"
            elif prob <= 0.4:
                bg_color = "#4488CC"  # 浅蓝 - 中等置信度
                text = f"模型建议: 可能没中  ({prob:.1%})"
            else:
                bg_color = "#88AACC"  # 更浅蓝 - 低置信度
                text = f"模型建议: 倾向没中  ({prob:.1%})"
            fg_color = "white"

        self.prediction_label.config(text=text, bg=bg_color, fg=fg_color)

    def update_info(self):
        if self.info_label:
            info_text = f"""进度：{self.current_index}/{len(self.image_files)}
训练集 - 有目标：{self.stats['train_target']} | 无目标：{self.stats['train_notarget']} | 已删除：{self.stats['deleted']}
当前文件：{os.path.basename(self.image_files[self.current_index]) if self.current_index < len(self.image_files) else 'N/A'}"""
            self.info_label.config(text=info_text)

    def load_image(self, image_path):
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size

            try:
                window_width = self.root.winfo_width()
                window_height = self.root.winfo_height()

                if window_width < 100 or window_height < 200:
                    window_width = 520
                    window_height = 700

                scale = min((window_width - 50) / img_width,
                           (window_height - 280) / img_height,
                           1.0)

                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            except Exception:
                pass

            self.current_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_image, text="")

        except Exception as e:
            print(f"  加载图片失败 {image_path}: {e}")
            self.image_label.config(image="", text=f"加载失败: {os.path.basename(image_path)}")

    def annotate_image(self, has_target):
        if self.current_index >= len(self.image_files):
            self.quit_program()
            return

        current_file = self.image_files[self.current_index]
        target_dir = self.train_target_dir if has_target else self.train_notarget_dir
        stat_key = "train_target" if has_target else "train_notarget"

        try:
            filename = os.path.basename(current_file)
            target_path = os.path.join(target_dir, filename)
            shutil.move(current_file, target_path)

            operation = {
                'type': 'annotate',
                'filename': filename,
                'source_path': current_file,
                'target_path': target_path,
                'has_target': has_target,
                'stat_key': stat_key
            }
            self.undo_stack.append(operation)
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

            self.stats[stat_key] += 1
            self.stats["total_processed"] += 1
            print(f"  {filename} -> {'有目标' if has_target else '无目标'}")

        except Exception as e:
            print(f"  移动文件失败: {e}")
            return

        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.load_image(self.image_files[self.current_index])
            self.update_prediction_display()
        else:
            print("  所有图片已标注完成!")
            self.quit_program()
            return

        self.update_info()

    def delete_image(self):
        if self.current_index >= len(self.image_files):
            self.quit_program()
            return

        current_file = self.image_files[self.current_index]
        filename = os.path.basename(current_file)

        recycle_dir = os.path.join(self.target_base, ".recycle")
        if not os.path.exists(recycle_dir):
            os.makedirs(recycle_dir, exist_ok=True)

        try:
            recycle_path = os.path.join(recycle_dir, filename)
            if os.path.exists(recycle_path):
                import time
                name, ext = os.path.splitext(filename)
                timestamp = int(time.time() * 1000)
                recycle_path = os.path.join(recycle_dir, f"{name}_{timestamp}{ext}")

            shutil.move(current_file, recycle_path)

            operation = {
                'type': 'delete',
                'filename': filename,
                'source_path': current_file,
                'recycle_path': recycle_path,
            }
            self.undo_stack.append(operation)
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

            self.stats['deleted'] += 1
            self.stats['total_processed'] += 1
            print(f"  已删除: {filename}")

        except Exception as e:
            print(f"  删除文件失败: {e}")
            return

        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.load_image(self.image_files[self.current_index])
            self.update_prediction_display()
        else:
            print("  所有图片已标注完成!")
            self.quit_program()
            return

        self.update_info()

    def undo_last_annotation(self):
        if not self.undo_stack:
            print("  没有可回退的操作")
            return

        last_operation = self.undo_stack.pop()

        try:
            if last_operation['type'] == 'delete':
                shutil.move(last_operation['recycle_path'], last_operation['source_path'])
                self.stats['deleted'] -= 1
                self.stats['total_processed'] -= 1
                print(f"  已恢复删除: {last_operation['filename']}")

            elif last_operation['type'] == 'annotate':
                shutil.move(last_operation['target_path'], last_operation['source_path'])
                self.stats[last_operation['stat_key']] -= 1
                self.stats["total_processed"] -= 1
                print(f"  已回退标注: {last_operation['filename']}")

            if self.current_index > 0:
                self.current_index -= 1

            if self.current_index < len(self.image_files):
                self.load_image(self.image_files[self.current_index])
                self.update_prediction_display()
            self.update_info()

        except Exception as e:
            print(f"  回退失败: {e}")
            self.undo_stack.append(last_operation)

    def quit_program(self):
        print("\n" + "=" * 50)
        print(f"训练集 - 有目标：{self.stats['train_target']} | 无目标：{self.stats['train_notarget']}")
        print(f"已删除：{self.stats['deleted']} 张 | 总计：{self.stats['total_processed']} 张")
        print("=" * 50)

        if self.root:
            self.root.quit()
            self.root.destroy()
        sys.exit(0)

    def run(self):
        print("=" * 50)
        print("数据标注助手")
        print("=" * 50)

        self.image_files = self.get_image_files()
        if not self.image_files:
            print("  未找到图片文件")
            return

        # 启动时预推理所有图片
        print("\n预推理:")
        self.load_model_and_predict(self.image_files)
        print()

        self.setup_gui()

        def show_first_image():
            try:
                if self.current_index < len(self.image_files):
                    self.load_image(self.image_files[self.current_index])
                    self.update_prediction_display()
                self.update_info()
            except Exception as e:
                print(f"  显示第一张图片失败: {e}")

        self.root.after(300, show_first_image)
        self.root.mainloop()


def main():
    helper = SelectHelper()
    try:
        helper.run()
    except KeyboardInterrupt:
        print("\n  用户中断程序")
        helper.quit_program()
    except Exception as e:
        print(f"\n  程序异常: {e}")
        helper.quit_program()

if __name__ == "__main__":
    main()
