#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiHelper - 激光射蚊子助手
多线程版本：实时截图推理 + 判断模式控制
内存优化版本
"""

import os
import sys
import time
import threading
import numpy as np
from PIL import Image
import mss
import cv2
import tensorflow as tf
from tensorflow import keras
import win32gui
import ctypes
import uuid
import ctypes.wintypes as wintypes
import winsound
from pynput.keyboard import Controller as KeyboardController
import gc
from datetime import datetime

# 全局变量及锁
global_lock = threading.Lock()  # 全局变量保护锁
if_exit_goal = 0  # 是否退出判断模式（1=退出，0=继续）
if_dead = 0       # 是否已经被其他机关击落（1=已击落，0=未击落）

class OptimizedInferenceModule:
    """优化的推理模块，基于inference-plus.py - 使用SavedModel格式"""
    def __init__(self, model_path="models-v0.8/best_model.h5", threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.img_height = 144
        self.img_width = 144

        # 设置随机种子保证可重现性（与train_model保持一致）
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # 复用对象以减少内存分配 - 初始化为None
        self._reuse_processed_image = None
        
        self._configure_optimizations() # 配置优化设置
        self.load_and_convert_model() # 加载并转换模型
        self._warmup_model() # 预热模型 

    def _configure_optimizations(self):
        """配置优化设置（与train_model保持一致）"""
        # GPU内存动态分配配置（与train_model保持一致）
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

    def load_and_convert_model(self):
        """加载H5模型并转换为SavedModel格式进行推理"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                sys.exit(1)
            
            # 生成SavedModel路径（与H5文件在同一目录）
            model_dir = os.path.dirname(self.model_path)
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            savedmodel_path = os.path.join(model_dir, f"{model_name}_savedmodel")
            
            # 检查SavedModel是否已存在
            if os.path.exists(savedmodel_path):
                print(f"📦 发现已存在的SavedModel: {savedmodel_path}")
                print("📦 直接加载SavedModel进行推理...")
                self.model = tf.saved_model.load(savedmodel_path)
                self.inference_func = self.model.signatures['serving_default']
                print("✅ SavedModel加载完成！")
                return
            
            print(f"📦 加载H5模型: {self.model_path}")
            
            # 第一步：加载H5模型
            h5_model = keras.models.load_model(self.model_path, compile=False)
            
            # 第二步：转换为SavedModel格式并保存到同目录
            print("🔄 转换H5模型为SavedModel格式...")
            h5_model.save(savedmodel_path, save_format='tf')
            
            # 第三步：加载SavedModel进行推理
            print("📦 加载SavedModel进行推理...")
            self.model = tf.saved_model.load(savedmodel_path)
            
            # 获取推理函数（SavedModel的默认签名）
            self.inference_func = self.model.signatures['serving_default']
            
            # 清理H5模型
            del h5_model
            gc.collect()
            
            print(f"✅ SavedModel转换和加载完成！已保存到: {savedmodel_path}")

        except Exception as e:
            print(f"❌ 模型转换失败: {e}")
            sys.exit(1)

    def _warmup_model(self):
        """预热SavedModel"""
        print("🔥 预热SavedModel...")
        dummy_image = None
        processed = None
        try:
            for _ in range(3):
                dummy_image = np.random.randint(0, 255, (self.img_height, self.img_width, 3), dtype=np.uint8)
                processed = self._fast_preprocess(dummy_image)
                
                # 使用SavedModel进行推理预热
                input_tensor = tf.constant(processed, dtype=tf.float32)
                self.inference_func(input_tensor)
                
                # 清理临时变量
                del dummy_image
                dummy_image = None
                gc.collect()  # 强制垃圾回收
        finally:
            # 确保清理资源
            if dummy_image is not None:
                del dummy_image
            if processed is not None:
                del processed
            gc.collect()
        print("✓ SavedModel预热完成")

    def _fast_preprocess(self, image_array):
        """优化的快速预处理"""
        resized = None
        img_array = None
        try:
            if len(image_array.shape) == 3:
                # PIL图像已经是RGB格式，直接resize
                resized = cv2.resize(image_array, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
            else:
                resized = cv2.resize(image_array, (self.img_width, self.img_height))

            img_array = resized.astype(np.float32) * (1.0/255.0)
            
            # 复用处理后的图像数组以减少内存分配
            if self._reuse_processed_image is None:
                self._reuse_processed_image = np.expand_dims(img_array, axis=0)
            else:
                # 直接更新现有数组的数据
                self._reuse_processed_image[0] = img_array
            
            return self._reuse_processed_image
        finally:
            # 清理临时变量
            if resized is not None and resized is not image_array:
                del resized
            if img_array is not None:
                del img_array

    def predict_from_pil_image(self, pil_image):
        """从PIL图像进行SavedModel推理"""
        image_array = None
        processed_image = None
        prediction = None
        input_tensor = None
        try:
            # 将PIL图像转换为numpy数组
            image_array = np.array(pil_image)
            processed_image = self._fast_preprocess(image_array)
            
            # 使用SavedModel进行推理
            input_tensor = tf.constant(processed_image, dtype=tf.float32)
            prediction = self.inference_func(input_tensor)
            
            # 提取概率值（SavedModel输出格式可能不同）
            if 'output_0' in prediction:
                probability = float(prediction['output_0'][0][0])
            elif 'dense_1' in prediction:
                probability = float(prediction['dense_1'][0][0])
            else:
                # 尝试获取第一个输出
                output_key = list(prediction.keys())[0]
                probability = float(prediction[output_key][0][0])
            
            return probability
        except Exception as e:
            print(f"❌ SavedModel推理失败: {e}")
            return 0.0
        finally:
            # 显式清理临时变量
            if image_array is not None:
                del image_array
            if prediction is not None:
                del prediction
            if input_tensor is not None:
                del input_tensor
            # 不删除processed_image因为它是复用的

class ScreenshotInferenceThread(threading.Thread):
    """截图推理线程"""
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module
        self.running = True
        self.screenshot_lock = threading.Lock()
        self.current_screenshot = None
        # 预计算截图区域
        self._precompute_capture_region()
        self._last_probability = 0.0
        self._gc_counter = 0  # 垃圾回收计数器
        
        # 帧率统计
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_interval = 5.0  # 5秒打印一次帧率

    def _precompute_capture_region(self, size=144):
        """预计算截图区域坐标"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            center_x = monitor["width"] // 2
            center_y = monitor["height"] // 2

            left = center_x - size // 2
            top = center_y - size // 2
            right = left + size
            bottom = top + size

            left = max(0, left)
            top = max(0, top)
            right = min(monitor["width"], right)
            bottom = min(monitor["height"], bottom)

            self.capture_region = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top
            }

    def run(self):
        """后台持续截图并推理"""
        sct = mss.mss()

        try:
            while self.running:
                screenshot = None
                img = None
                try:
                    # 截取屏幕中心区域
                    screenshot = sct.grab(self.capture_region)
                    
                    # 每次都创建新的PIL图像，避免复用导致的问题
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                    # 使用锁保护current_screenshot
                    with self.screenshot_lock:
                        # 显式释放之前的图像对象以避免内存泄漏
                        if self.current_screenshot is not None:
                            try:
                                self.current_screenshot.close()
                            except:
                                pass  # 忽略关闭错误
                        self.current_screenshot = img.copy()  # 创建副本避免引用问题

                    # 进行推理
                    probability = self.inference_module.predict_from_pil_image(img)

                    # 帧率统计
                    self._frame_count += 1
                    current_time = time.time()
                    if current_time - self._last_fps_time >= self._fps_interval:
                        fps = self._frame_count / (current_time - self._last_fps_time)
                        print(f"📊 平均帧率: {fps:.1f} FPS (过去{self._fps_interval:.0f}秒处理了{self._frame_count}帧)")
                        self._frame_count = 0
                        self._last_fps_time = current_time

                    # 根据推理结果设置全局变量（使用锁保护）
                    global if_exit_goal
                    with global_lock:
                        old_value = if_exit_goal
                        if_exit_goal = 1 if probability > 0.5 else 0

                    # 调试信息：只在值改变时打印
                    if if_exit_goal != old_value:
                        print(f"🎯 推理结果更新: 概率={probability:.3f}, if_exit_goal={if_exit_goal}")

                    # 定期进行垃圾回收（每100次推理）
                    self._gc_counter += 1
                    if self._gc_counter >= 100:
                        gc.collect()
                        self._gc_counter = 0

                    # 每轮增加5ms延迟，降低CPU占用率
                    time.sleep(0.01)

                except Exception as e:
                    print(f"截图推理线程出错: {e}")
                    time.sleep(0.1)
                finally:
                    # 确保资源被释放
                    if img is not None:
                        try:
                            img.close()
                        except:
                            pass
                    if screenshot is not None:
                        del screenshot
                    
        finally:
            try:
                sct.close()
            except:
                pass

    def get_current_screenshot(self):
        """获取当前截图"""
        with self.screenshot_lock:
            if self.current_screenshot is not None:
                return self.current_screenshot.copy()
            return None

    def stop(self):
        """停止线程并清理资源"""
        self.running = False
        # 清理当前截图资源
        with self.screenshot_lock:
            if self.current_screenshot is not None:
                try:
                    self.current_screenshot.close()
                except:
                    pass
                self.current_screenshot = None
        # 强制垃圾回收
        gc.collect()

# --- Windows API 结构体定义 ---
user32 = ctypes.windll.user32

RIDEV_INPUTSINK = 0x00000100
RIDEV_NOLEGACY = 0x00000030
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0

WM_INPUT = 0x00FF

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("ulButtons", wintypes.ULONG),
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]

class RAWINPUTUNION(ctypes.Union):
    _fields_ = [
        ("mouse", RAWMOUSE),
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", RAWINPUTUNION),
    ]

# --- Raw Input 鼠标监听器 ---
class RawInputMouseListener:
    def __init__(self, on_click_callback):
        self.on_click_callback = on_click_callback
        self.running = True
        self.thread = threading.Thread(target=self._message_loop, daemon=True)
        # 保存Windows资源引用用于清理
        self.hwnd = None
        self.class_atom = None
        self.thread.start()

    def _message_loop(self):
        try:
            # 创建隐藏窗口
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._wnd_proc
            wc.lpszClassName = f"RawInputListener_{uuid.uuid4()}"
            hinst = wc.hInstance = win32gui.GetModuleHandle(None)
            class_atom = win32gui.RegisterClass(wc)
            hwnd = win32gui.CreateWindow(class_atom, "RawInputHidden", 0, 0, 0, 0, 0, 0, 0, hinst, None)

            if not hwnd:
                raise RuntimeError("窗口创建失败")

            # 保存资源引用用于清理
            self.hwnd = hwnd
            self.class_atom = class_atom

            # 注册 Raw Input 鼠标 - 尝试不同的注册模式
            rid = RAWINPUTDEVICE()
            rid.usUsagePage = 0x01  # 通用桌面控制
            rid.usUsage = 0x02      # 鼠标
            rid.dwFlags = RIDEV_INPUTSINK  # 全局捕获所有应用程序的输入
            rid.hwndTarget = hwnd

            print(f"🔧 尝试注册模式1: flags=0x{rid.dwFlags:08X}, hwnd={rid.hwndTarget}")

            if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                print("⚠️ 模式1失败，尝试模式2: 不指定窗口")
                rid.hwndTarget = None
                if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                    print("⚠️ 模式2失败，尝试模式3: RIDEV_NOLEGACY")
                    rid.dwFlags = RIDEV_INPUTSINK | RIDEV_NOLEGACY
                    rid.hwndTarget = hwnd
                    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                        print("⚠️ 模式3失败，尝试模式4: 仅RIDEV_NOLEGACY")
                        rid.dwFlags = RIDEV_NOLEGACY
                        if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                            raise RuntimeError("所有Raw Input注册模式都失败")
            print("✅ Raw Input注册成功")

            # 消息循环
            while self.running:
                win32gui.PumpWaitingMessages()
                time.sleep(0.01)

        except Exception as e:
            print(f"❌ 鼠标监听器初始化失败: {e}")
            raise
        finally:
            # 清理Windows资源
            try:
                if self.hwnd:
                    win32gui.DestroyWindow(self.hwnd)
                    self.hwnd = None
            except Exception as e:
                print(f"⚠️ 清理窗口资源时出错: {e}")

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            self._handle_raw_input(lparam)
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _handle_raw_input(self, lparam):
        buf = None
        try:
            size = wintypes.UINT()
            user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))

            if size.value == 0:
                print("❌ Raw Input数据大小为0")
                return

            buf = ctypes.create_string_buffer(size.value)
            result = user32.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))

            if result == -1:
                print("❌ 获取Raw Input数据失败")
                return

            # 直接从原始数据包解析按键信息（第28字节）
            button_flags = int.from_bytes(buf.raw[28:32], byteorder='little', signed=False)

            # 只处理按键事件（非0值）
            if button_flags != 0:
                if button_flags == 0x01:  # 左键按下
                    self.on_click_callback('left', True)
                elif button_flags == 0x04:  # 右键按下
                    self.on_click_callback('right', True)

        except Exception as e:
            print(f"❌ 处理Raw Input数据失败: {e}")
        finally:
            # 清理临时缓冲区
            if buf is not None:
                del buf

    def stop(self):
        self.running = False

        # 等待线程结束
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # 清理Windows资源
        try:
            if self.hwnd:
                win32gui.DestroyWindow(self.hwnd)
                self.hwnd = None
            if self.class_atom:
                win32gui.UnregisterClass(self.class_atom, None)
                self.class_atom = None
        except Exception as e:
            print(f"⚠️ 清理Raw Input资源时出错: {e}")


class FeedbackCollector:
    """反馈数据收集器 - 用于收集开火时的截图数据"""
    def __init__(self, save_dir="image"):
        self.save_dir = save_dir
        self.feedback_count = 0
        
        # 确保保存目录存在（如果不存在才创建，避免覆盖现有目录）
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"✓ 创建反馈数据保存目录: {self.save_dir}")
        else:
            print(f"✓ 使用现有目录保存反馈数据: {self.save_dir}")
    
    def collect_feedback_image(self, pil_image, probability):
        """收集反馈图像数据并立即保存到磁盘"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"feedback_{timestamp}_{self.feedback_count:04d}_prob{probability:.3f}.png"
            
            # 直接保存到磁盘
            filepath = os.path.join(self.save_dir, filename)
            pil_image.save(filepath, format='PNG')
            
            print(f"✓ 反馈图像已保存: {filename}")
            self.feedback_count += 1
            return True
            
        except Exception as e:
            print(f"反馈数据保存失败: {e}")
            return False
    

class WiHelper:
    """激光射蚊子助手主类"""
    def __init__(self):
        self.judging_mode = False  # 判断模式状态
        self.judging_start_time = 0  # 判断模式开始时间
        self.right_mouse_pressed = False  # 右键按下状态
        self.left_mouse_pressed = False   # 左键按下状态
        self.judging_thread = None  # 判断模式线程
        self.judging_lock = threading.Lock()  # 判断模式状态锁
        self.mouse_listener = RawInputMouseListener(self.on_mouse_click)

        # 初始化推理模块
        self.inference_module = OptimizedInferenceModule()

        # 初始化截图推理线程
        self.screenshot_thread = ScreenshotInferenceThread(self.inference_module)
        self.screenshot_thread.start()
        
        # 初始化反馈数据收集器
        self.feedback_collector = FeedbackCollector()

        # 设置进程伪装
        try:
            ctypes.windll.kernel32.SetConsoleTitleW("Windows Service Host")
            print("✓ 进程名称伪装完成")
        except Exception as e:
            print(f"✗ 进程名称伪装失败: {e}")

        # 内存监控计数器
        self._memory_check_counter = 0

    def on_mouse_click(self, button, pressed):
        """鼠标点击事件处理"""
        global if_dead, global_lock

        try:
            print(f"📡 收到鼠标事件: {button} {'按下' if pressed else '释放'}")  # 临时调试

            if button == 'right':
                self.right_mouse_pressed = pressed
                if pressed:
                    # 使用非阻塞方式检查状态
                    if self.judging_mode:
                        # 在判断模式中按右键，退出判断模式
                        print("🖱️ 右键点击：退出判断模式")
                        self.exit_judging_mode()
                    else:
                        # 不在判断模式，直接启动判断线程
                        if self.judging_thread is None or not self.judging_thread.is_alive():
                            print("🖱️ 右键点击：启动判断模式")
                            self.judging_thread = threading.Thread(target=self.enter_judging_mode_sync, daemon=True)
                            self.judging_thread.start()
                        else:
                            print("🖱️ 右键点击：判断线程正在运行中")
            elif button == 'left':
                self.left_mouse_pressed = pressed
                if pressed and self.judging_mode:
                    # 在判断模式中按左键，退出判断模式
                    print("🖱️ 左键点击：退出判断模式")
                    self.exit_judging_mode()
        except Exception as e:
            print(f"❌ 鼠标事件处理出错: {e}")

    def enter_judging_mode_sync(self):
        """同步进入判断模式（在当前线程中执行）"""
        global if_dead, if_exit_goal, global_lock

        with global_lock:
            if if_dead == 1:
                print("⚠️ 目标已被其他机关击落，跳过判断")
                return

        with self.judging_lock:
            self.judging_mode = True
            self.judging_start_time = time.time()
            print("🎯 进入瞄准模式 - 炮台正在瞄准中...")

        # 记录进入判断模式时的鼠标状态，避免立即退出
        initial_left_pressed = self.left_mouse_pressed
        initial_right_pressed = self.right_mouse_pressed

        aiming_time = 0.5  # 炮台瞄准时间
        total_timeout = 4.0  # 总超时时间
        start_time = time.time()

        try:
            # 第一阶段：0.5秒炮台瞄准时间，只检测退出条件，不开火
            while time.time() - start_time < aiming_time:
                # 检查瞄准模式是否被外部退出（非阻塞检查）
                if not self.judging_mode:
                    print("🖱️ 瞄准模式被外部退出")
                    return

                # 检查退出条件：只有在判断期间新按下的鼠标键才算退出
                if (self.left_mouse_pressed and not initial_left_pressed) or \
                   (self.right_mouse_pressed and not initial_right_pressed):
                    print("🖱️ 检测到瞄准期间的鼠标点击，退出瞄准模式")
                    self.exit_judging_mode()
                    return

                # 在瞄准时间内不检查目标，避免立即开火
                # 调试：每100ms打印一次当前状态
                current_time = time.time()
                if not hasattr(self, '_last_debug_time'):
                    self._last_debug_time = 0
                if current_time - self._last_debug_time > 0.1:
                    with global_lock:
                        current_if_exit_goal = if_exit_goal
                    print(f"🔍 炮台瞄准中... if_exit_goal={current_if_exit_goal}, 经过时间={current_time - start_time:.1f}s")
                    self._last_debug_time = current_time

                time.sleep(0.01)  # 小延迟避免CPU占用过高

            # 第二阶段：继续等待直到3秒超时，但仍然实时检测
            while time.time() - start_time < total_timeout:
                # 检查瞄准模式是否被外部退出（非阻塞检查）
                if not self.judging_mode:
                    print("🖱️ 瞄准模式被外部退出")
                    return

                # 检查退出条件
                if (self.left_mouse_pressed and not initial_left_pressed) or \
                   (self.right_mouse_pressed and not initial_right_pressed):
                    print("🖱️ 检测到瞄准期间的鼠标点击，退出瞄准模式")
                    self.exit_judging_mode()
                    return

                # 实时检查目标
                with global_lock:
                    current_if_exit_goal = if_exit_goal
                if current_if_exit_goal == 1:
                    print("🎯 检测到目标，开火！")
                    self.fire_laser()
                    self.exit_judging_mode()
                    return

                time.sleep(0.01)  # 小延迟避免CPU占用过高

            # 超时未检测到目标
            print("❌ 判断超时，未检测到有效目标")
            self.exit_judging_mode()
        finally:
            # 确保退出判断模式
            self.exit_judging_mode()

    def exit_judging_mode(self):
        """退出判断模式"""
        with self.judging_lock:
            self.judging_mode = False
            self.judging_start_time = 0
            print("🏁 退出判断模式")

    def fire_laser(self):
        """开火 - 模拟键盘敲击P并播放音乐，同时收集反馈数据"""
        keyboard = None
        try:
            # 收集反馈数据 - 获取当前截图和推理概率
            current_screenshot = self.screenshot_thread.get_current_screenshot()
            if current_screenshot is not None:
                # 重新进行推理获取准确概率（而不是使用二值化的if_exit_goal）
                current_probability = self.inference_module.predict_from_pil_image(current_screenshot)
                
                # 收集反馈数据
                self.feedback_collector.collect_feedback_image(current_screenshot, current_probability)
                print(f"📊 已收集反馈数据，概率: {current_probability:.3f}")
            else:
                print("⚠️ 无法获取当前截图，跳过反馈数据收集")
            
            # 使用pynput模拟键盘敲击P
            keyboard = KeyboardController()
            keyboard.press('p')
            keyboard.release('p')

            # 播放开火音乐
            self.play_fire_sound()
            print("💥 激光发射成功！")
            # 确保退出判断模式
            self.exit_judging_mode()

        except Exception as e:
            print(f"❌ 开火失败: {e}")
        finally:
            if keyboard is not None:
                del keyboard

    def play_fire_sound(self):
        """播放开火音效"""
        try:
            # 使用系统默认提示音
            winsound.MessageBeep(0x40)
        except Exception as e:
            print(f"❌ 音频播放失败: {e}")

    def run(self):
        """主循环"""
        print("🚀 WiHelper激光射蚊子助手启动")
        print("右键点击进入判断模式")
        print("每次开火时会自动收集反馈数据并立即保存到image文件夹")
        print("按Ctrl+C退出")

        try:
            # 主线程只负责监听鼠标事件，不再阻塞
            while True:
                # 定期进行垃圾回收
                self._memory_check_counter += 1
                if self._memory_check_counter >= 1000:  # 每1000次循环进行一次垃圾回收
                    gc.collect()
                    self._memory_check_counter = 0
                
                time.sleep(0.1)  # 定期检查监听器状态

        except KeyboardInterrupt:
            print("\n👋 用户中断")
            self.cleanup()
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")

        # 停止判断线程
        if self.judging_thread and self.judging_thread.is_alive():
            self.judging_thread.join(timeout=1.0)

        # 停止截图推理线程
        if self.screenshot_thread:
            self.screenshot_thread.stop()
            self.screenshot_thread.join(timeout=1.0)

        # 停止鼠标监听器
        if self.mouse_listener:
            self.mouse_listener.stop()

        # 清理TensorFlow资源
        try:
            tf.keras.backend.clear_session()
            print("✓ TensorFlow会话已清理")
        except Exception as e:
            print(f"⚠️ TensorFlow清理失败: {e}")

        # 强制垃圾回收
        gc.collect()
        print("✅ 清理完成")

def main():
    # 创建WiHelper实例并运行
    helper = WiHelper()
    helper.run()

if __name__ == "__main__":
    main()