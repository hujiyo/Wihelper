import os
import time
import threading
import uuid
from datetime import datetime
from PIL import Image
import mss
import keyboard as kb
from pynput import mouse, keyboard as pynput_kb
import ctypes
import ctypes.wintypes as wintypes
import win32gui
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("警告：winsound模块不可用，将使用视觉反馈")

from config import AppConfig, ScreenshotConfig


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


class RawInputMouseListener:
    """基于 Raw Input API 的鼠标监听器（不安装低级钩子，降低被反作弊检测的风险）"""
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

            if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                rid.hwndTarget = None
                if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                    rid.dwFlags = RIDEV_INPUTSINK | RIDEV_NOLEGACY
                    rid.hwndTarget = hwnd
                    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                        rid.dwFlags = RIDEV_NOLEGACY
                        if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                            raise RuntimeError("所有Raw Input注册模式都失败")

            while self.running:
                win32gui.PumpWaitingMessages()
                time.sleep(0.005)

        except Exception as e:
            print(f"Raw Input 鼠标监听器初始化失败: {e}")
            raise
        finally:
            try:
                if self.hwnd:
                    win32gui.DestroyWindow(self.hwnd)
                    self.hwnd = None
            except Exception as e:
                print(f"清理窗口资源时出错: {e}")

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

            if result == -1:
                return

            button_flags = int.from_bytes(buf.raw[28:32], byteorder='little', signed=False)

            if button_flags != 0:
                if button_flags == 0x01:
                    self.on_click_callback('left', True)
                elif button_flags == 0x04:
                    self.on_click_callback('right', True)

        except Exception as e:
            print(f"处理Raw Input数据失败: {e}")
        finally:
            if buf is not None:
                del buf

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        try:
            if self.hwnd:
                win32gui.DestroyWindow(self.hwnd)
                self.hwnd = None
            if self.class_atom:
                win32gui.UnregisterClass(self.class_atom, None)
                self.class_atom = None
        except Exception as e:
            print(f"清理Raw Input资源时出错: {e}")


class ScreenshotCollector:
    def __init__(self, save_dir=ScreenshotConfig.SAVE_DIR):
        self.save_dir = save_dir
        self.image_count = 0
        self.left_alt_pressed = False  # 跟踪左Alt键状态

        # 实时截图相关属性
        self.last_save_time = 0  # 上次保存时间
        self.save_cooldown = ScreenshotConfig.SAVE_COOLDOWN  # 保存冷却时间（秒）
        self.current_screenshot = None  # 当前截图
        self.screenshot_lock = threading.Lock()  # 保护current_screenshot的锁
        self.running = True  # 控制后台线程运行的标志

    def _precompute_capture_region(self, size=ScreenshotConfig.RAW_CAPTURE_SIZE):
        """预计算截图区域坐标，避免每次重新计算"""
        # 使用临时mss实例获取屏幕信息
        with mss.mss() as sct:
            # 获取主屏幕信息
            monitor = sct.monitors[0]  # 主屏幕

            # 计算中心区域坐标
            center_x = monitor["width"] // 2
            center_y = monitor["height"] // 2

            # 计算截图区域
            left = center_x - size // 2
            top = center_y - size // 2
            right = left + size
            bottom = top + size

            # 确保区域在屏幕范围内
            left = max(0, left)
            top = max(0, top)
            right = min(monitor["width"], right)
            bottom = min(monitor["height"], bottom)

            # 保存预计算的截图区域
            self.capture_region = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top
            }

    def _background_capture(self):
        """后台持续截图线程"""
        # 在后台线程中创建mss实例，避免线程本地存储问题
        sct = mss.mss()

        try:
            while self.running:
                try:
                    # 截取屏幕中心 raw_size x raw_size 区域
                    img = self.capture_center_region_thread_safe(sct, ScreenshotConfig.RAW_CAPTURE_SIZE)
                    # 处理图像
                    img = self.process_image(img)

                    # 使用锁保护current_screenshot
                    with self.screenshot_lock:
                        self.current_screenshot = img

                    # 短暂休眠，避免占用过多CPU
                    time.sleep(ScreenshotConfig.BACKGROUND_CAPTURE_INTERVAL)  # 每20ms截图一次

                except Exception as e:
                    print(f"后台截图出错: {e}")
                    time.sleep(0.1)
        finally:
            # 清理mss实例
            if hasattr(sct, 'close'):
                sct.close()

    def capture_center_region_thread_safe(self, sct, size=ScreenshotConfig.RAW_CAPTURE_SIZE):
        """线程安全的截取屏幕中心指定大小的方形区域"""
        # 使用传入的mss实例和预计算的截图区域
        screenshot = sct.grab(self.capture_region)

        # 转换为PIL图像
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        return img

    def process_image(self, img):
        """处理图像为3通道8比特格式"""
        # 确保图像是RGB模式（3通道）
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 确保图像是8比特深度（PIL默认就是8比特）
        return img

    def save_image(self, img):
        """直接保存图像到磁盘（不加密）"""
        timestamp = datetime.now().strftime(ScreenshotConfig.FILENAME_TIMESTAMP_FORMAT)
        filename = ScreenshotConfig.FILENAME_PATTERN.format(
            timestamp=timestamp, index=self.image_count
        )
        filepath = os.path.join(self.save_dir, filename)

        try:
            img.save(filepath, format='PNG')
            self.image_count += 1
            print(f"✓ 截图已保存: {filename}")
        except Exception as e:
            print(f"✗ 截图保存失败 {filename}: {e}")

    def play_success_sound(self):
        """播放成功提示音"""
        if AUDIO_AVAILABLE:
            try:
                winsound.MessageBeep(AppConfig.SOUND_SUCCESS)
            except Exception as e:
                print(f"音频播放失败: {e}")
                self.show_visual_feedback("✅ 截图成功!")
        else:
            self.show_visual_feedback("✅ 截图成功!")

    def play_error_sound(self):
        """播放错误提示音"""
        if AUDIO_AVAILABLE:
            try:
                winsound.MessageBeep(AppConfig.SOUND_ERROR)
            except Exception as e:
                print(f"音频播放失败: {e}")
                self.show_visual_feedback("❌ 截图失败!")
        else:
            self.show_visual_feedback("❌ 截图失败!")

    def show_visual_feedback(self, message):
        """显示视觉反馈"""
        print(f"\n{'='*50}")
        print(f"🎯 {message}")
        print(f"{'='*50}\n")

    def on_press(self, key):
        """键盘按下监听回调"""
        try:
            if key == pynput_kb.Key.alt_l:
                self.left_alt_pressed = True
        except AttributeError:
            pass

    def on_release(self, key):
        """键盘释放监听回调"""
        try:
            if key == pynput_kb.Key.alt_l:
                self.left_alt_pressed = False
        except AttributeError:
            pass

    def on_click(self, button, pressed):
        """鼠标点击监听回调"""
        try:
            if button == 'left' and pressed:
                # 检查是否按住左Alt键（防误触）
                if self.left_alt_pressed:
                    return  # 忽略这次点击

                # 检查冷却时间
                current_time = time.time()
                if current_time - self.last_save_time < self.save_cooldown:
                    remaining_time = self.save_cooldown - (current_time - self.last_save_time)
                    print(f"冷却中，还需等待 {remaining_time:.1f} 秒")
                    return  # 在冷却时间内，忽略保存

                # 获取当前截图
                with self.screenshot_lock:
                    if self.current_screenshot is None:
                        print("当前没有可用的截图")
                        return
                    img = self.current_screenshot.copy()

                # 保存图像
                self.save_image(img)
                self.last_save_time = current_time
                # 播放成功提示音
                self.play_success_sound()
        except Exception as e:
            print(f"截图过程中出错: {e}")
            # 播放错误提示音
            self.play_error_sound()

    def start(self):
        """启动截图收集器"""
        # 预计算截图区域
        self._precompute_capture_region()

        # 确保保存目录存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 启动后台截图线程
        self.screenshot_thread = threading.Thread(target=self._background_capture, daemon=True)
        self.screenshot_thread.start()

        print("=== 自动截图工具已启动 ===")
        print(f"保存目录: {self.save_dir}")
        print("实时截图模式：后台持续截图")
        print("左键保存截图，左键+Alt忽略")
        print("保存冷却时间：1秒")
        print("✓ 鼠标监听使用 Raw Input 模式")
        print("按Ctrl+C退出")

        # 创建鼠标监听器（Raw Input 模式，不安装低级钩子）
        self.mouse_listener = RawInputMouseListener(self.on_click)

        # 创建键盘监听器
        self.keyboard_listener = pynput_kb.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

        try:
            # 使用循环代替join，让Ctrl+C更容易中断
            while self.running and self.mouse_listener.thread.is_alive():
                time.sleep(0.1)  # 短暂休眠，避免占用太多CPU
        except KeyboardInterrupt:
            print("\n收到中断信号，正在退出...")
            self.stop()
        except Exception as e:
            print(f"监听过程中出错: {e}")
            self.stop()

    def stop(self):
        """停止监听"""
        print("程序退出...")

        # 首先停止后台截图线程
        self.running = False
        if hasattr(self, 'screenshot_thread'):
            self.screenshot_thread.join(timeout=1.0)

        kb.unhook_all()
        if hasattr(self, 'mouse_listener'):
            self.mouse_listener.stop()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        # 关闭 mss 实例
        if hasattr(self, 'sct'):
            self.sct.close()
        os._exit(0)

def main():
    # 创建截图收集器
    collector = ScreenshotCollector()

    try:
        # 启动监听
        collector.start()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        collector.stop()
    except Exception as e:
        print(f"发生错误: {e}")
        collector.stop()

if __name__ == "__main__":
    main()
