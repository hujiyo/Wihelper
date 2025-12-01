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
- 简单直接：有文件就处理，没文件就结束
"""

import os, sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import shutil

class SelectHelper:
    def __init__(self, source_dir="image", target_base="image"):
        self.source_dir = source_dir
        self.target_base = target_base

        # 创建目标文件夹
        self.train_target_dir = os.path.join(target_base, "train", "got")
        self.train_notarget_dir = os.path.join(target_base, "train", "nogot")

        for dir_path in [self.train_target_dir, self.train_notarget_dir]:
            if os.path.exists(dir_path):
                # 检查文件夹是否为空
                existing_files = os.listdir(dir_path)
                if existing_files:
                    print(f"📁 文件夹已存在且包含 {len(existing_files)} 个文件: {dir_path}")
                else:
                    print(f"📁 文件夹已存在（空文件夹）: {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                print(f"📁 创建新文件夹: {dir_path}")

        # 获取所有图片文件
        self.image_files = []
        self.current_index = 0

        # 统计信息
        self.stats = {
            "train_target": 0,
            "train_notarget": 0,
            "deleted": 0,
            "total_processed": 0
        }

        # 回退功能
        self.undo_stack = []  # 存储最近的操作，最多5个
        self.max_undo = 5

        # GUI相关
        self.root = None
        self.image_label = None
        self.info_label = None
        self.current_image = None

    def get_image_files(self):
        """获取所有图片文件"""
        if not os.path.exists(self.source_dir):
            print(f"❌ 源文件夹不存在: {self.source_dir}")
            return []

        all_files = []
        try:
            # 只处理当前目录下的文件，不包括子文件夹
            for file in os.listdir(self.source_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(self.source_dir, file)
                    # 确保是文件而不是文件夹
                    if os.path.isfile(full_path):
                        all_files.append(full_path)
        except Exception as e:
            print(f"❌ 读取目录失败: {e}")
            return []

        # 按文件名排序
        all_files.sort()
        print(f"📁 找到 {len(all_files)} 张图片文件")
        return all_files

    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("自动开火工具 - 图片标注助手")
        self.root.geometry("500x600")
        self.root.resizable(True, True)

        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="🎯数据标注助手",font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # 图片显示区域
        self.image_label = ttk.Label(main_frame, text="加载中...",background="lightgray")
        self.image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 信息显示
        self.info_label = ttk.Label(main_frame, text="准备开始...",font=("Arial", 10))
        self.info_label.grid(row=2, column=0, pady=(0, 10))

        # 操作说明
        instructions = """
操作说明：
• 空格键：没中 (notarget)
• 回车键：中了 (target)
• Delete键：删除图片（质量不高）
• Ctrl+Z：回退上一次操作 (最多5步)
• Ctrl+C：退出程序

标注模式：训练集
        """
        instr_label = ttk.Label(main_frame, text=instructions,font=("Arial", 9), justify=tk.LEFT)
        instr_label.grid(row=3, column=0, pady=(0, 10))

        # 绑定键盘事件
        self.root.bind('<space>', lambda e: self.annotate_image(False))
        self.root.bind('<Return>', lambda e: self.annotate_image(True))
        self.root.bind('<Delete>', lambda e: self.delete_image())
        self.root.bind('<Control-z>', lambda e: self.undo_last_annotation())

        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.quit_program)

    def update_info(self):
        """更新信息显示"""
        if self.info_label:
            info_text = f"""进度：{self.current_index}/{len(self.image_files)}
训练集 - 有目标：{self.stats['train_target']} | 无目标：{self.stats['train_notarget']} | 已删除：{self.stats['deleted']}
当前文件：{os.path.basename(self.image_files[self.current_index]) if self.current_index < len(self.image_files) else 'N/A'}"""
            self.info_label.config(text=info_text)

    def load_image(self, image_path):
        """加载并显示图片"""
        try:
            # 打开图片
            img = Image.open(image_path)
            img_width, img_height = img.size

            # 获取窗口大小，添加安全检查
            try:
                window_width = self.root.winfo_width()
                window_height = self.root.winfo_height()

                # 如果窗口大小不合理，使用默认值
                if window_width < 100 or window_height < 200:
                    window_width = 500
                    window_height = 600

                # 计算缩放比例
                scale = min((window_width - 50) / img_width,
                           (window_height - 200) / img_height,
                           1.0)  # 不放大图片

                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            except Exception as scale_error:
                print(f"⚠️ 缩放计算失败，使用原始尺寸: {scale_error}")
                # 如果缩放失败，使用原始尺寸
                pass

            # 转换为PhotoImage
            self.current_image = ImageTk.PhotoImage(img)

            # 显示图片
            self.image_label.config(image=self.current_image, text="")

        except Exception as e:
            print(f"❌ 加载图片失败 {image_path}: {e}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.image_label.config(image="", text=f"加载失败: {os.path.basename(image_path)}")

    def annotate_image(self, has_target):
        """标注图片"""
        if self.current_index >= len(self.image_files):
            print("🎉 所有图片已标注完成！")
            self.quit_program()
            return

        current_file = self.image_files[self.current_index]

        # 确定目标文件夹（只使用训练集）
        target_dir = self.train_target_dir if has_target else self.train_notarget_dir
        stat_key = "train_target" if has_target else "train_notarget"

        # 移动文件（从原始文件夹移动到分类文件夹）
        try:
            filename = os.path.basename(current_file)
            target_path = os.path.join(target_dir, filename)
            shutil.move(current_file, target_path)  # 移动文件，原始文件会被删除

            # 记录操作到回退栈
            operation = {
                'type': 'annotate',
                'filename': filename,
                'source_path': current_file,
                'target_path': target_path,
                'has_target': has_target,
                'stat_key': stat_key
            }
            self.undo_stack.append(operation)

            # 保持回退栈大小
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

            # 更新统计
            self.stats[stat_key] += 1
            self.stats["total_processed"] += 1

            print(f"✅ 标注完成: {filename} -> {'有目标' if has_target else '无目标'} (训练集)")

        except Exception as e:
            print(f"❌ 移动文件失败: {e}")
            return

        # 移动到下一张
        self.current_index += 1

        # 显示下一张图片
        if self.current_index < len(self.image_files):
            self.load_image(self.image_files[self.current_index])
        else:
            print("🎉 所有图片已标注完成！")
            self.quit_program()
            return

        self.update_info()

    def delete_image(self):
        """删除当前图片"""
        if self.current_index >= len(self.image_files):
            print("🎉 所有图片已标注完成！")
            self.quit_program()
            return

        current_file = self.image_files[self.current_index]
        filename = os.path.basename(current_file)

        # 创建回收站文件夹（用于临时存储删除的文件，支持回退）
        recycle_dir = os.path.join(self.target_base, ".recycle")
        if not os.path.exists(recycle_dir):
            os.makedirs(recycle_dir, exist_ok=True)

        try:
            # 移动文件到回收站而不是直接删除
            recycle_path = os.path.join(recycle_dir, filename)
            
            # 如果回收站中已存在同名文件，添加时间戳
            if os.path.exists(recycle_path):
                import time
                name, ext = os.path.splitext(filename)
                timestamp = int(time.time() * 1000)
                recycle_path = os.path.join(recycle_dir, f"{name}_{timestamp}{ext}")
            
            shutil.move(current_file, recycle_path)

            # 记录操作到回退栈（删除操作现在可以回退）
            operation = {
                'type': 'delete',
                'filename': filename,
                'source_path': current_file,
                'recycle_path': recycle_path,
            }
            self.undo_stack.append(operation)

            # 保持回退栈大小
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

            # 更新统计
            self.stats['deleted'] += 1
            self.stats['total_processed'] += 1

            print(f"🗑️ 已删除: {filename}")

        except Exception as e:
            print(f"❌ 删除文件失败: {e}")
            return

        # 移动到下一张
        self.current_index += 1

        # 显示下一张图片
        if self.current_index < len(self.image_files):
            self.load_image(self.image_files[self.current_index])
        else:
            print("🎉 所有图片已标注完成！")
            self.quit_program()
            return

        self.update_info()

    def undo_last_annotation(self):
        """回退上一次操作"""
        if not self.undo_stack:
            print("ℹ️ 没有可回退的操作")
            return

        # 获取最后一次操作
        last_operation = self.undo_stack.pop()

        try:
            # 检查操作类型
            if last_operation['type'] == 'delete':
                # 回退删除操作：从回收站恢复文件
                shutil.move(last_operation['recycle_path'], last_operation['source_path'])
                
                # 调整统计信息
                self.stats['deleted'] -= 1
                self.stats['total_processed'] -= 1
                
                print(f"↩️ 已恢复删除: {last_operation['filename']}")
                
            elif last_operation['type'] == 'annotate':
                # 回退标注操作：将文件移回源文件夹
                shutil.move(last_operation['target_path'], last_operation['source_path'])

                # 调整统计信息
                self.stats[last_operation['stat_key']] -= 1
                self.stats["total_processed"] -= 1

                print(f"↩️ 已回退标注: {last_operation['filename']}")

            # 调整当前索引
            if self.current_index > 0:
                self.current_index -= 1

            # 重新显示当前图片
            if self.current_index < len(self.image_files):
                self.load_image(self.image_files[self.current_index])
            self.update_info()

        except Exception as e:
            print(f"❌ 回退失败: {e}")
            # 如果回退失败，将操作重新放回栈中
            self.undo_stack.append(last_operation)

    def quit_program(self):
        """退出程序"""
        print("\n" + "="*50)
        print("📊 标注统计：")
        print(f"训练集 - 有目标：{self.stats['train_target']} | 无目标：{self.stats['train_notarget']}")
        print(f"已删除：{self.stats['deleted']} 张")
        print(f"总计处理：{self.stats['total_processed']} 张图片")
        print("="*50)

        if self.root:
            self.root.quit()
            self.root.destroy()
        sys.exit(0)

    def run(self):
        print("="*60)
        print("🎯 数据标注助手")
        print("="*60)
        print("操作说明：")
        print("• 空格键：无目标 (nogot)")
        print("• 回车键：有目标 (got)")
        print("• Delete键：删除图片（质量不高）")
        print("• Ctrl+Z：回退上一次操作 (最多5步)")
        print("• Ctrl+C：退出程序")
        print("="*60)

        # 获取图片文件
        self.image_files = self.get_image_files()
        if not self.image_files:
            print("❌ 未找到任何图片文件，标注任务已完成！")
            return

        # 设置GUI
        self.setup_gui()

        # 延迟显示第一张图片，确保窗口完全加载
        def show_first_image():
            try:
                if self.current_index < len(self.image_files):
                    self.load_image(self.image_files[self.current_index])
                self.update_info()
            except Exception as e:
                print(f"⚠️ 显示第一张图片失败: {e}")

        # 延迟500ms显示图片
        self.root.after(500, show_first_image)

        # 启动GUI主循环
        self.root.mainloop()

def main():
    helper = SelectHelper()
    try:
        helper.run()
    except KeyboardInterrupt:
        print("\n👋 用户中断程序")
        helper.quit_program()
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        helper.quit_program()

if __name__ == "__main__":
    main()
