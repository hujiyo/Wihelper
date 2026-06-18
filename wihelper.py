#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiHelper - 激光射蚊子助手
多线程版本：实时截图推理 + 判断模式控制
使用 ONNX Runtime 推理，无需 Python + PyTorch 环境
"""

import os
import sys

# 在 numpy 导入前设置，避免 MKL 多线程 DLL 缺失导致崩溃
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import time
import threading
import configparser
import numpy as np
from PIL import Image
import mss
import win32gui
import ctypes
import uuid
import ctypes.wintypes as wintypes
import winsound
from pynput.keyboard import Controller as KeyboardController
import gc
import signal
from datetime import datetime

from config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    PathsConfig,
    ScreenshotConfig,
)


# --- 配置文件 ---
def get_base_dir():
    """获取程序所在目录（兼容 PyInstaller 打包后的路径）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    """加载配置文件，不存在则创建默认配置；默认值来自 AppConfig。"""
    base_dir = get_base_dir()
    config_path = os.path.join(base_dir, AppConfig.CONFIG_INI_FILENAME)

    config = configparser.ConfigParser()
    section = AppConfig.CONFIG_INI_SECTION

    # 默认值统一从 AppConfig 读取
    defaults = {
        'threshold': str(AppConfig.DEFAULT_FIRE_THRESHOLD),
        'fire_cooldown': str(AppConfig.DEFAULT_FIRE_COOLDOWN),
        'model_path': AppConfig.DEFAULT_ONNX_MODEL_PATH,
        'target_fps': str(AppConfig.DEFAULT_TARGET_FPS),
    }

    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
        if section not in config:
            config[section] = defaults
        else:
            for key, default_val in defaults.items():
                if key not in config[section]:
                    config[section][key] = default_val
    else:
        config[section] = defaults
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                config.write(f)
            print(f"✓ 已创建默认配置文件: {config_path}")
        except Exception as e:
            print(f"⚠️ 创建配置文件失败: {e}")

    return config[section]


# 全局变量及锁
global_lock = threading.Lock()
if_exit_goal = 0
if_dead = 0
current_result = 0

class OptimizedInferenceModule:
    """ONNX Runtime 推理模块 - 必须使用 GPU（DirectML / CUDA），禁止 CPU 兜底"""
    def __init__(self, model_path=AppConfig.DEFAULT_ONNX_MODEL_PATH, threshold=AppConfig.DEFAULT_FIRE_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.capture_size = AppConfig.CAPTURE_SIZE
        # FP32 默认；FP16 模型加载后会被 _probe_io 改成 np.float16
        self._input_dtype = np.float32

        # 预分配推理输入 buffer，避免每帧重新分配
        self._input_buf = np.zeros(
            (1, 3, AppConfig.CAPTURE_SIZE, AppConfig.CAPTURE_SIZE), dtype=np.float32
        )
        self._inv_255 = np.float32(1.0 / 255.0)
        self._is_gpu = False
        self._load_model()
        self._warmup_model()

    def _load_model(self):
        """加载 ONNX 模型"""
        import onnxruntime as ort

        base_dir = get_base_dir()
        abs_model_path = os.path.join(base_dir, self.model_path)
        if not os.path.exists(abs_model_path):
            if os.path.exists(self.model_path):
                abs_model_path = self.model_path
            else:
                print(f"❌ 模型文件不存在: {abs_model_path}")
                print(f"   也未找到: {self.model_path}")
                sys.exit(1)

        self.model_path = abs_model_path
        print(f"🚀 加载 ONNX 模型: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # DirectML 官方要求：必须关闭 memory pattern 和使用顺序执行模式
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # GPU 推理时 CPU 线程数越少越好，减少线程争抢
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        # 必须使用 GPU：DML 优先 → CUDA 次之
        available_providers = ort.get_available_providers()
        providers = []
        if 'DmlExecutionProvider' in available_providers:
            providers.append('DmlExecutionProvider')
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if not providers:
            print("❌ 未检测到任何 GPU 执行提供者 (DML / CUDA)")
            sys.exit(1)

        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_options, providers=providers
        )

        active_provider = self.session.get_providers()[0]
        self._is_gpu = active_provider in ('DmlExecutionProvider', 'CUDAExecutionProvider')
        provider_name = {
            'DmlExecutionProvider': 'DirectML (GPU)',
            'CUDAExecutionProvider': 'CUDA (GPU)',
        }.get(active_provider, active_provider)

        print(f"✅ 模型加载完成！执行提供者: {provider_name}")

        # 自动探测：输入 dtype (FP32/FP16)
        self._probe_io()

    def _probe_io(self):
        """探测 ONNX 模型的输入 dtype（FP32 / FP16）"""
        try:
            import onnx
            onnx_model = onnx.load(self.model_path)
            input_type = onnx_model.graph.input[0].type.tensor_type.elem_type
            # 1 = FLOAT, 10 = FLOAT16
            if input_type == 10:
                self._input_dtype = np.float16
                # 重建 buffer 为 fp16
                self._input_buf = np.zeros(
                    (1, 3, AppConfig.CAPTURE_SIZE, AppConfig.CAPTURE_SIZE),
                    dtype=np.float16,
                )
                self._inv_255 = np.float16(1.0 / 255.0)
                print("   输入精度: FP16（GPU 推理将提速）")
            else:
                print("   输入精度: FP32")
        except Exception as e:
            print(f"   ⚠️ IO 探测失败: {e}，按默认 FP32 处理")

    def _warmup_model(self):
        """预热模型"""
        print("🔥 预热模型...")
        try:
            dummy_bgra = np.random.randint(
                0, 255,
                (AppConfig.CAPTURE_SIZE, AppConfig.CAPTURE_SIZE, AppConfig.WARMUP_DUMMY_CHANNELS),
                dtype=np.uint8,
            ).tobytes()
            for _ in range(AppConfig.WARMUP_RUNS):
                _ = self.predict_from_raw_bgra(
                    dummy_bgra, AppConfig.CAPTURE_SIZE, AppConfig.CAPTURE_SIZE
                )
        except Exception as e:
            print(f"⚠️ 预热出错: {e}")
        print("✓ 模型预热完成")

    def predict_from_raw_bgra(self, bgra_bytes, width, height):
        """从 mss 截图的原始 BGRA 字节直接推理（零 PIL，零中间数组）"""
        try:
            raw = np.frombuffer(bgra_bytes, dtype=np.uint8).reshape(height, width, 4)
            buf = self._input_buf[0]  # shape (3, H, W)
            # BGRA → RGB，直接写入预分配的 CHW buffer
            buf[0] = raw[:, :, 2]  # R
            buf[1] = raw[:, :, 1]  # G
            buf[2] = raw[:, :, 0]  # B
            # in-place 归一化（dtype 自动跟随 buffer：fp16/fp32）
            buf *= self._inv_255
            # 推理 + sigmoid → 概率
            logit = self.session.run(None, {ModelConfig.INPUT_NAME: self._input_buf})[0]
            return float(1.0 / (1.0 + np.exp(-logit[0, 0])))
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return 0.0

    def predict_from_pil_image(self, pil_image):
        """从 PIL 图像推理（兼容反馈截图等场景）"""
        try:
            img = np.array(pil_image, dtype=self._input_dtype)
            img *= np.array(self._inv_255, dtype=self._input_dtype)
            buf = self._input_buf[0]
            buf[0] = img[:, :, 0]  # R
            buf[1] = img[:, :, 1]  # G
            buf[2] = img[:, :, 2]  # B
            logit = self.session.run(None, {ModelConfig.INPUT_NAME: self._input_buf})[0]
            return float(1.0 / (1.0 + np.exp(-logit[0, 0])))
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return 0.0

class ScreenshotInferenceThread(threading.Thread):
    """截图推理线程"""
    def __init__(self, inference_module, target_fps=AppConfig.DEFAULT_TARGET_FPS):
        super().__init__()
        self.inference_module = inference_module
        self.target_fps = max(
            AppConfig.MIN_FPS, min(target_fps, AppConfig.MAX_FPS)
        )  # 限制在 MIN_FPS~MAX_FPS
        self.running = True
        self.screenshot_lock = threading.Lock()
        self.current_screenshot = None
        self._precompute_capture_region()
        self._last_probability = 0.0
        self._gc_counter = 0

        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_interval = AppConfig.FPS_STAT_INTERVAL

    def _precompute_capture_region(self):
        size = self.inference_module.capture_size

        with mss.MSS() as sct:
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
        sct = mss.MSS()
        target_frame_time = 1.0 / self.target_fps

        try:
            while self.running:
                frame_start = time.perf_counter()
                screenshot = None
                try:
                    screenshot = sct.grab(self.capture_region)

                    probability = self.inference_module.predict_from_raw_bgra(
                        screenshot.bgra, screenshot.width, screenshot.height
                    )

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
                    # BitBlt 等间歇性截图错误静默跳过，不影响运行
                    pass
                finally:
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
RIM_TYPEKEYBOARD = 1

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

class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]

class RAWINPUTUNION(ctypes.Union):
    _fields_ = [
        ("mouse", RAWMOUSE),
        ("keyboard", RAWKEYBOARD),
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
            self.thread.join(timeout=AppConfig.MOUSE_LISTENER_JOIN_TIMEOUT)
        try:
            if self.hwnd:
                win32gui.DestroyWindow(self.hwnd)
                self.hwnd = None
            if self.class_atom:
                win32gui.UnregisterClass(self.class_atom, None)
                self.class_atom = None
        except Exception as e:
            print(f"⚠️ 清理Raw Input资源时出错: {e}")


# --- Raw Input 键盘监听器 ---
class RawInputKeyboardListener:
    """Raw Input 键盘监听器

    与 RawInputMouseListener 相互独立：各自的窗口、各自的消息循环、各自的 stop 方法。
    共用 RAWINPUT / RAWINPUTHEADER / RAWINPUTUNION 结构体定义。
    监听 F 键（VK_F = 0x46）的按下事件。
    """

    # VK_F 是 Windows 头文件 winuser.h 硬编码的标准虚拟键码
    VK_F = 0x46
    # RI_KEY_MAKE (按位与 1 == 0) 表示按下事件；与 E0/E1 前缀标志位并存
    RI_KEY_BREAK_MASK = 0x01
    # HID UsagePage/Usage 规范定义：Generic Desktop / Keyboard
    HID_USAGE_PAGE_GENERIC = 0x01
    HID_USAGE_KEYBOARD = 0x06

    def __init__(self, on_key_press_callback):
        self.on_key_press_callback = on_key_press_callback
        self.running = True
        self.hwnd = None
        self.class_atom = None
        self._self_check_done = False
        self.thread = threading.Thread(target=self._message_loop, daemon=True)
        self.thread.start()

    def _message_loop(self):
        try:
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._wnd_proc
            wc.lpszClassName = f"RawInputKeyboard_{uuid.uuid4()}"
            hinst = wc.hInstance = win32gui.GetModuleHandle(None)
            class_atom = win32gui.RegisterClass(wc)
            hwnd = win32gui.CreateWindow(class_atom, "RawInputKeyboardHidden", 0, 0, 0, 0, 0, 0, 0, hinst, None)

            if not hwnd:
                raise RuntimeError("键盘监听窗口创建失败")

            self.hwnd = hwnd
            self.class_atom = class_atom

            rid = RAWINPUTDEVICE()
            rid.usUsagePage = self.HID_USAGE_PAGE_GENERIC
            rid.usUsage = self.HID_USAGE_KEYBOARD
            rid.dwFlags = RIDEV_INPUTSINK
            rid.hwndTarget = hwnd

            print(f"🔧 尝试注册键盘 Raw Input 模式1: flags=0x{rid.dwFlags:08X}, hwnd={rid.hwndTarget}")

            if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                print("⚠️ 模式1失败，尝试模式2: 不指定窗口")
                rid.hwndTarget = None
                if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                    print("⚠️ 模式2失败，尝试模式3: RIDEV_INPUTSINK | RIDEV_NOLEGACY")
                    rid.dwFlags = RIDEV_INPUTSINK | RIDEV_NOLEGACY
                    rid.hwndTarget = hwnd
                    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                        print("⚠️ 模式3失败，尝试模式4: 仅RIDEV_NOLEGACY")
                        rid.dwFlags = RIDEV_NOLEGACY
                        if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                            raise RuntimeError("所有键盘 Raw Input 注册模式都失败")
            print("✅ 键盘 Raw Input 注册成功（按 F 键验证链路）")

            while self.running:
                win32gui.PumpWaitingMessages()
                time.sleep(0.005)

        except Exception as e:
            print(f"❌ 键盘监听器初始化失败: {e}")
            raise
        finally:
            try:
                if self.hwnd:
                    win32gui.DestroyWindow(self.hwnd)
                    self.hwnd = None
            except Exception as e:
                print(f"⚠️ 清理键盘窗口资源时出错: {e}")

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
                return

            buf = ctypes.create_string_buffer(size.value)
            result = user32.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))

            if result == -1 or result != size.value:
                return

            raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents

            if raw.header.dwType != RIM_TYPEKEYBOARD:
                return

            kb = raw.data.keyboard
            is_f_press = kb.VKey == self.VK_F and (kb.Flags & self.RI_KEY_BREAK_MASK) == 0

            # 首次按 F 键时打印一次自检通过，避免静默失败
            if is_f_press and not self._self_check_done:
                print(f"⌨️ 收到 F 键 Raw Input 事件 (VKey=0x{kb.VKey:02X}, Flags=0x{kb.Flags:04X})")
                self._self_check_done = True

            # 仅响应 F 键按下（忽略抬起、忽略 E0/E1 之外的修饰位）
            if is_f_press:
                self.on_key_press_callback()

        except Exception as e:
            print(f"❌ 处理键盘 Raw Input 数据失败: {e}")
        finally:
            if buf is not None:
                del buf

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=AppConfig.MOUSE_LISTENER_JOIN_TIMEOUT)
        try:
            if self.hwnd:
                win32gui.DestroyWindow(self.hwnd)
                self.hwnd = None
            if self.class_atom:
                win32gui.UnregisterClass(self.class_atom, None)
                self.class_atom = None
        except Exception as e:
            print(f"⚠️ 清理键盘 Raw Input 资源时出错: {e}")


class FeedbackCollector:
    """反馈数据收集器"""
    def __init__(self, save_dir=DataConfig.DATA_DIR):
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
    def __init__(
        self,
        fire_cooldown=AppConfig.DEFAULT_FIRE_COOLDOWN,
        model_path=AppConfig.DEFAULT_ONNX_MODEL_PATH,
        threshold=AppConfig.DEFAULT_FIRE_THRESHOLD,
        target_fps=AppConfig.DEFAULT_TARGET_FPS,
    ):
        self.judging_mode = False
        self.judging_start_time = 0
        self.right_mouse_pressed = False
        self.left_mouse_pressed = False
        self.f_key_pressed = False
        self.judging_thread = None
        self.judging_lock = threading.Lock()
        self.mouse_listener = RawInputMouseListener(self.on_mouse_click)

        self.keyboard_listener = RawInputKeyboardListener(self.on_key_press)

        self.fire_cooldown = fire_cooldown

        self.inference_module = OptimizedInferenceModule(
            model_path=model_path, threshold=threshold
        )
        self.screenshot_thread = ScreenshotInferenceThread(self.inference_module, target_fps=target_fps)
        self.screenshot_thread.start()

        self.feedback_collector = FeedbackCollector()

        try:
            ctypes.windll.kernel32.SetConsoleTitleW(AppConfig.CONSOLE_TITLE)
            print("✓ 进程名称伪装完成")
        except Exception as e:
            print(f"✗ 进程名称伪装失败: {e}")

        self._memory_check_counter = 0

    def on_key_press(self):
        # F 键事件已在 RawInputKeyboardListener 内部过滤，这里只关心按下后的业务逻辑
        if self.judging_mode:
            print("⌨️ F键按下：打断判断模式")
            self.f_key_pressed = True

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

        aiming_time = AppConfig.AIMING_TIME
        total_timeout = AppConfig.TOTAL_TIMEOUT
        start_time = time.time()

        fire_count = 0
        max_fire_count = AppConfig.MAX_FIRE_COUNT
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
                if current_time - self._last_debug_time > AppConfig.DEBUG_INTERVAL:
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

                        is_sniper_mode = self.fire_cooldown >= AppConfig.SNIPER_COOLDOWN_THRESHOLD
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
            if self.fire_cooldown >= AppConfig.SNIPER_COOLDOWN_THRESHOLD:
                current_screenshot = self.screenshot_thread.get_current_screenshot()
                if current_screenshot is not None:
                    current_probability = self.inference_module.predict_from_pil_image(current_screenshot)
                    self.feedback_collector.collect_feedback_image(current_screenshot, current_probability)
                    print(f"📊 已收集反馈数据，概率: {current_probability:.3f}")
                else:
                    print("⚠️ 无法获取当前截图，跳过反馈数据收集")

            keyboard = KeyboardController()
            keyboard.press(AppConfig.FIRE_KEY)
            keyboard.release(AppConfig.FIRE_KEY)
            print("💥 激光发射成功！")

        except Exception as e:
            print(f"❌ 开火失败: {e}")
        finally:
            if keyboard is not None:
                del keyboard

    def play_fire_sound(self):
        try:
            winsound.MessageBeep(AppConfig.SOUND_SUCCESS)
        except Exception as e:
            print(f"❌ 音频播放失败: {e}")

    def run(self):
        print("🚀 WiHelper激光射蚊子助手启动")
        print(f"⏱️  当前开枪延迟设置: {self.fire_cooldown}秒")
        if self.fire_cooldown >= AppConfig.SNIPER_COOLDOWN_THRESHOLD:
            print(f"📍 模式: 大狙模式（单发精确射击，{AppConfig.SNIPER_COOLDOWN_THRESHOLD}秒延迟相当于一枪后自动退出）")
            print("💾 反馈数据: 每次开火时会自动保存截图到image文件夹")
        else:
            print(f"📍 模式: 连狙模式（{self.fire_cooldown}秒延迟，{AppConfig.TOTAL_TIMEOUT}秒内最多{AppConfig.MAX_FIRE_COUNT}枪）")
            print("💾 反馈数据: 连狙模式不保存截图")
        print("🖱️  右键点击进入判断模式")
        print(f"⌨️  按{AppConfig.INTERRUPT_KEY.upper()}键可打断判断模式（左键不会打断）")
        print("⌨️  按Ctrl+C退出程序")

        try:
            while True:
                self._memory_check_counter += 1
                if self._memory_check_counter >= AppConfig.MEMORY_CHECK_INTERVAL:
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
            self.judging_thread.join(timeout=AppConfig.THREAD_JOIN_TIMEOUT)

        if self.screenshot_thread:
            self.screenshot_thread.stop()
            self.screenshot_thread.join(timeout=AppConfig.THREAD_JOIN_TIMEOUT)

        if self.mouse_listener:
            self.mouse_listener.stop()

        if self.keyboard_listener:
            self.keyboard_listener.stop()

        gc.collect()
        print("✅ 清理完成")

def main():
    # 优先注册SIGINT处理器，防止Intel Fortran运行时拦截Ctrl+C
    def _sigint_handler(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    print("=" * 60)
    print("🎯 WiHelper 激光射蚊子助手")
    print("=" * 60)

    # 加载配置
    config = load_config()
    threshold = float(config['threshold'])
    fire_cooldown = float(config['fire_cooldown'])
    model_path = config['model_path']
    target_fps = int(config['target_fps'])

    print(f"⚙️  配置:")
    print(f"   模型路径: {model_path}")
    print(f"   检测阈值: {threshold}")
    print(f"   开枪延迟: {fire_cooldown}秒")
    print(f"   目标帧率: {target_fps} FPS")
    print()

    helper = WiHelper(
        fire_cooldown=fire_cooldown,
        model_path=model_path,
        threshold=threshold,
        target_fps=target_fps,
    )
    helper.run()

if __name__ == "__main__":
    main()
