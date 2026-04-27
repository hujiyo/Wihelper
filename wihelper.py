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
import torch
import win32gui
import ctypes
import uuid
import ctypes.wintypes as wintypes
import winsound
from pynput.keyboard import Controller as KeyboardController, Listener as KeyboardListener, Key
import gc
from datetime import datetime
from train_model import WiHelperCNN


# 全局变量及锁
global_lock = threading.Lock()
if_exit_goal = 0
if_dead = 0
current_result = 0

class OptimizedInferenceModule:
    """优化的推理模块 - 直接使用120×120输入"""
    def __init__(self, model_path="models-v1.1-4/best_model.pth", threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold

        self.capture_size = 120
        self.img_height = 120
        self.img_width = 120

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 加载模型: {model_path}")
        print(f"   设备: {self.device}")

        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self._reuse_buffer = None
        self.load_model()
        self._warmup_model()

    def load_model(self):
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                sys.exit(1)

            print(f"📦 加载模型: {self.model_path}")
            self.model = WiHelperCNN()
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ 模型加载完成！")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def _warmup_model(self):
        """预热模型"""
        print("🔥 预热模型...")
        try:
            for _ in range(3):
                dummy_image = np.random.randint(0, 255, (self.img_height, self.img_width, 3), dtype=np.uint8)
                processed = self._fast_preprocess(dummy_image)
                with torch.no_grad():
                    self.model(processed)
                del dummy_image
        except Exception as e:
            print(f"⚠️ 预热过程中出错: {e}")
        print("✓ 模型预热完成")

    def _fast_preprocess(self, image_array):
        """优化的快速预处理"""
        img_float = image_array.astype(np.float32) * (1.0 / 255.0)
        # HWC -> CHW
        img_chw = np.transpose(img_float, (2, 0, 1))

        if self._reuse_buffer is None:
            self._reuse_buffer = np.expand_dims(img_chw, axis=0)
        else:
            self._reuse_buffer[0] = img_chw

        return torch.from_numpy(self._reuse_buffer).to(self.device)

    def predict_from_pil_image(self, pil_image):
        """从PIL图像进行推理"""
        try:
            image_array = np.array(pil_image)
            processed_image = self._fast_preprocess(image_array)

            with torch.no_grad():
                output = self.model(processed_image)
                probability = torch.sigmoid(output).item()

            return probability
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return 0.0

class ScreenshotInferenceThread(threading.Thread):
    """截图推理线程"""
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module
        self.running = True
        self.screenshot_lock = threading.Lock()
        self.current_screenshot = None
        self._precompute_capture_region()
        self._last_probability = 0.0
        self._gc_counter = 0

        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_interval = 5.0

    def _precompute_capture_region(self):
        size = self.inference_module.capture_size

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
        sct = mss.mss()
        target_frame_time = 1.0 / 60.0  # 目标60FPS

        try:
            while self.running:
                frame_start = time.perf_counter()
                screenshot = None
                img = None
                try:
                    screenshot = sct.grab(self.capture_region)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                    probability = self.inference_module.predict_from_pil_image(img)

                    self._frame_count += 1
                    current_time = time.time()
                    if current_time - self._last_fps_time >= self._fps_interval:
                        fps = self._frame_count / (current_time - self._last_fps_time)
                        print(f"📊 平均帧率: {fps:.1f} FPS (过去{self._fps_interval:.0f}秒处理了{self._frame_count}帧)")
                        self._frame_count = 0
                        self._last_fps_time = current_time

                    global if_exit_goal, current_result
                    with global_lock:
                        old_value = if_exit_goal
                        old_current = current_result

                        current_result = 1 if probability > self.inference_module.threshold else 0
                        if_exit_goal = 1 if probability > self.inference_module.threshold else 0

                    if if_exit_goal != old_value or current_result != old_current:
                        print(f"🎯 推理结果更新: 概率={probability:.3f}, current={current_result}, if_exit_goal={if_exit_goal}")

                    # 帧率限制：等待剩余时间
                    elapsed = time.perf_counter() - frame_start
                    sleep_time = target_frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    print(f"截图推理线程出错: {e}")
                finally:
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
        with self.screenshot_lock:
            if self.current_screenshot is not None:
                return self.current_screenshot.copy()
            return None

    def stop(self):
        self.running = False
        with self.screenshot_lock:
            if self.current_screenshot is not None:
                try:
                    self.current_screenshot.close()
                except:
                    pass
                self.current_screenshot = None
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
        self.hwnd = None
        self.class_atom = None
        self.thread.start()

    def _message_loop(self):
        try:
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._wnd_proc
            wc.lpszClassName = f"RawInputListener_{uuid.uuid4()}"
            hinst = wc.hInstance = win32gui.GetModuleHandle(None)
            class_atom = win32gui.RegisterClass(wc)
            hwnd = win32gui.CreateWindow(class_atom, "RawInputHidden", 0, 0, 0, 0, 0, 0, 0, hinst, None)

            if not hwnd:
                raise RuntimeError("窗口创建失败")

            self.hwnd = hwnd
            self.class_atom = class_atom

            rid = RAWINPUTDEVICE()
            rid.usUsagePage = 0x01
            rid.usUsage = 0x02
            rid.dwFlags = RIDEV_INPUTSINK
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

            while self.running:
                win32gui.PumpWaitingMessages()
                time.sleep(0.005)

        except Exception as e:
            print(f"❌ 鼠标监听器初始化失败: {e}")
            raise
        finally:
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

            button_flags = int.from_bytes(buf.raw[28:32], byteorder='little', signed=False)

            if button_flags != 0:
                if button_flags == 0x01:
                    self.on_click_callback('left', True)
                elif button_flags == 0x04:
                    self.on_click_callback('right', True)

        except Exception as e:
            print(f"❌ 处理Raw Input数据失败: {e}")
        finally:
            if buf is not None:
                del buf

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
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
    """反馈数据收集器"""
    def __init__(self, save_dir="image"):
        self.save_dir = save_dir
        self.feedback_count = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"✓ 创建反馈数据保存目录: {self.save_dir}")
        else:
            print(f"✓ 使用现有目录保存反馈数据: {self.save_dir}")

    def collect_feedback_image(self, pil_image, probability):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"feedback_{timestamp}_{self.feedback_count:04d}_prob{probability:.3f}.png"
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
    def __init__(self, fire_cooldown=4.0):
        self.judging_mode = False
        self.judging_start_time = 0
        self.right_mouse_pressed = False
        self.left_mouse_pressed = False
        self.f_key_pressed = False
        self.judging_thread = None
        self.judging_lock = threading.Lock()
        self.mouse_listener = RawInputMouseListener(self.on_mouse_click)

        self.keyboard_listener = KeyboardListener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self.fire_cooldown = fire_cooldown

        self.inference_module = OptimizedInferenceModule()
        self.screenshot_thread = ScreenshotInferenceThread(self.inference_module)
        self.screenshot_thread.start()

        self.feedback_collector = FeedbackCollector()

        try:
            ctypes.windll.kernel32.SetConsoleTitleW("Windows Service Host")
            print("✓ 进程名称伪装完成")
        except Exception as e:
            print(f"✗ 进程名称伪装失败: {e}")

        self._memory_check_counter = 0

    def on_key_press(self, key):
        try:
            if hasattr(key, 'char') and key.char == 'f':
                if self.judging_mode:
                    print("⌨️ F键按下：打断判断模式")
                    self.f_key_pressed = True
        except AttributeError:
            pass

    def on_mouse_click(self, button, pressed):
        global if_dead, global_lock

        try:
            print(f"📡 收到鼠标事件: {button} {'按下' if pressed else '释放'}")

            if button == 'right':
                self.right_mouse_pressed = pressed
                if pressed:
                    if self.judging_mode:
                        print("🖱️ 右键点击：退出判断模式")
                        self.exit_judging_mode()
                    else:
                        if self.judging_thread is None or not self.judging_thread.is_alive():
                            print("🖱️ 右键点击：启动判断模式")
                            self.judging_thread = threading.Thread(target=self.enter_judging_mode_sync, daemon=True)
                            self.judging_thread.start()
                        else:
                            print("🖱️ 右键点击：判断线程正在运行中")
            elif button == 'left':
                self.left_mouse_pressed = pressed
        except Exception as e:
            print(f"❌ 鼠标事件处理出错: {e}")

    def enter_judging_mode_sync(self):
        global if_dead, if_exit_goal, current_result, global_lock

        with global_lock:
            if if_dead == 1:
                print("⚠️ 目标已被其他机关击落，跳过判断")
                return

        with self.judging_lock:
            self.judging_mode = True
            self.judging_start_time = time.time()
            print("🎯 进入瞄准模式 - 炮台正在瞄准中...")

        initial_left_pressed = self.left_mouse_pressed
        initial_right_pressed = self.right_mouse_pressed

        aiming_time = 0.5
        total_timeout = 4.0
        start_time = time.time()

        fire_count = 0
        max_fire_count = 8
        last_fire_time = 0
        fire_cooldown = self.fire_cooldown

        try:
            while time.time() - start_time < aiming_time:
                if not self.judging_mode:
                    print("🖱️ 瞄准模式被外部退出")
                    return

                if (self.right_mouse_pressed and not initial_right_pressed) or self.f_key_pressed:
                    reason = "F键" if self.f_key_pressed else "右键"
                    print(f"🖱️ 检测到{reason}打断，退出瞄准模式")
                    self.f_key_pressed = False
                    self.exit_judging_mode()
                    return

                current_time = time.time()
                if not hasattr(self, '_last_debug_time'):
                    self._last_debug_time = 0
                if current_time - self._last_debug_time > 0.1:
                    with global_lock:
                        current_if_exit_goal = if_exit_goal
                        current_current_result = current_result
                    print(f"🔍 炮台瞄准中... current={current_current_result}, if_exit_goal={current_if_exit_goal}, 经过时间={current_time - start_time:.1f}s")
                    self._last_debug_time = current_time

                time.sleep(0.005)

            while time.time() - start_time < total_timeout:
                if not self.judging_mode:
                    print(f"🖱️ 瞄准模式被外部退出 (已开火{fire_count}枪)")
                    return

                if (self.right_mouse_pressed and not initial_right_pressed) or self.f_key_pressed:
                    reason = "F键" if self.f_key_pressed else "右键"
                    print(f"🖱️ 检测到{reason}打断，退出瞄准模式 (已开火{fire_count}枪)")
                    self.f_key_pressed = False
                    self.exit_judging_mode()
                    return

                with global_lock:
                    current_if_exit_goal = if_exit_goal
                    current_current_result = current_result

                current_time = time.time()

                if current_current_result == 1:
                    if current_time - last_fire_time >= fire_cooldown:
                        fire_count += 1
                        print(f"🎯 检测到目标，立即开火！(第{fire_count}/{max_fire_count}枪)")
                        self.fire_laser()
                        last_fire_time = current_time

                        is_sniper_mode = self.fire_cooldown >= 4.0
                        if fire_count >= max_fire_count or is_sniper_mode:
                            reason = f"达到最大开火次数({max_fire_count}枪)" if not is_sniper_mode else "大狙模式单发命中"
                            print(f"✅ {reason}，退出瞄准模式")
                            self.exit_judging_mode()
                            return

                time.sleep(0.005)

            if fire_count > 0:
                print(f"⏱️ 判断超时，共开火{fire_count}枪")
            else:
                print("❌ 判断超时，未检测到有效目标")
            self.exit_judging_mode()
        finally:
            self.exit_judging_mode()

    def exit_judging_mode(self):
        with self.judging_lock:
            self.judging_mode = False
            self.judging_start_time = 0
            print("🏁 退出判断模式")

    def fire_laser(self):
        keyboard = None
        try:
            if self.fire_cooldown >= 4.0:
                current_screenshot = self.screenshot_thread.get_current_screenshot()
                if current_screenshot is not None:
                    current_probability = self.inference_module.predict_from_pil_image(current_screenshot)
                    self.feedback_collector.collect_feedback_image(current_screenshot, current_probability)
                    print(f"📊 已收集反馈数据，概率: {current_probability:.3f}")
                else:
                    print("⚠️ 无法获取当前截图，跳过反馈数据收集")

            keyboard = KeyboardController()
            keyboard.press('p')
            keyboard.release('p')
            print("💥 激光发射成功！")

        except Exception as e:
            print(f"❌ 开火失败: {e}")
        finally:
            if keyboard is not None:
                del keyboard

    def play_fire_sound(self):
        try:
            winsound.MessageBeep(0x40)
        except Exception as e:
            print(f"❌ 音频播放失败: {e}")

    def run(self):
        print("🚀 WiHelper激光射蚊子助手启动")
        print(f"⏱️  当前开枪延迟设置: {self.fire_cooldown}秒")
        if self.fire_cooldown >= 4.0:
            print("📍 模式: 大狙模式（单发精确射击，4秒延迟相当于一枪后自动退出）")
            print("💾 反馈数据: 每次开火时会自动保存截图到image文件夹")
        else:
            print(f"📍 模式: 连狙模式（{self.fire_cooldown}秒延迟，4秒内最多8枪）")
            print("💾 反馈数据: 连狙模式不保存截图")
        print("🖱️  右键点击进入判断模式")
        print("⌨️  按F键可打断判断模式（左键不会打断）")
        print("⌨️  按Ctrl+C退出程序")

        try:
            while True:
                self._memory_check_counter += 1
                if self._memory_check_counter >= 1000:
                    gc.collect()
                    self._memory_check_counter = 0
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n👋 用户中断")
            self.cleanup()
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            self.cleanup()

    def cleanup(self):
        print("正在清理资源...")

        if self.judging_thread and self.judging_thread.is_alive():
            self.judging_thread.join(timeout=1.0)

        if self.screenshot_thread:
            self.screenshot_thread.stop()
            self.screenshot_thread.join(timeout=1.0)

        if self.mouse_listener:
            self.mouse_listener.stop()

        if self.keyboard_listener:
            self.keyboard_listener.stop()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ GPU缓存已清理")
        except Exception as e:
            print(f"⚠️ GPU清理失败: {e}")

        gc.collect()
        print("✅ 清理完成")

def main():
    import sys

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        print("\n✗ 未检测到CUDA GPU，无法继续！")
        print("  请确认:")
        print("  1. 已安装 NVIDIA 显卡驱动")
        print("  2. 已安装 CUDA Toolkit")
        print("  3. 已安装对应版本的 PyTorch")
        sys.exit(1)

    print("=" * 60)
    print("🎯 WiHelper激光射蚊子助手 - 启动配置")
    print("=" * 60)
    print()
    print("请设置开枪延迟时间（每一枪之间的时间间隔）：")
    print("  - 直接按回车: 使用默认4秒（大狙模式，一枪后自动退出瞄准）")
    print("  - 输入数字: 自定义延迟时间（例如：0.2 表示每0.2秒一枪，连狙模式）")
    print()
    print("💡 说明:")
    print("  - 延迟 >= 4秒: 大狙模式，第一枪后因为延迟长会超时退出")
    print("  - 延迟 < 4秒: 连狙模式，4秒内最多连续射击8枪")
    print()

    fire_cooldown = 4.0

    try:
        user_input = input("请输入延迟时间（秒）或直接回车使用默认值: ").strip()

        if user_input == "":
            print(f"✓ 使用默认延迟: {fire_cooldown}秒（大狙模式）")
        else:
            try:
                fire_cooldown = float(user_input)
                if fire_cooldown <= 0:
                    print("⚠️ 延迟时间必须大于0，使用默认值4秒")
                    fire_cooldown = 4.0
                elif fire_cooldown > 10:
                    print("⚠️ 延迟时间过长（>10秒），使用默认值4秒")
                    fire_cooldown = 4.0
                else:
                    if fire_cooldown >= 4.0:
                        print(f"✓ 已设置延迟: {fire_cooldown}秒（大狙模式）")
                    else:
                        print(f"✓ 已设置延迟: {fire_cooldown}秒（连狙模式）")
            except ValueError:
                print("⚠️ 输入格式错误，使用默认值4秒")
                fire_cooldown = 4.0
    except Exception as e:
        print(f"⚠️ 输入错误: {e}，使用默认值4秒")
        fire_cooldown = 4.0

    print()
    print("=" * 60)
    print()

    helper = WiHelper(fire_cooldown=fire_cooldown)
    helper.run()

if __name__ == "__main__":
    main()
