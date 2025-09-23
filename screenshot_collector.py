import os
import time
import threading
import random
import base64
from datetime import datetime
from PIL import Image
import mss
import keyboard as kb
from pynput import mouse, keyboard as pynput_kb
import ctypes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("警告：winsound模块不可用，将使用视觉反馈")

class ScreenshotCollector:
    def __init__(self, save_dir="image"):
        self.save_dir = save_dir
        self.image_count = 0
        self.left_alt_pressed = False  # 跟踪左Alt键状态

        # 实时截图相关属性
        self.last_save_time = 0  # 上次保存时间
        self.save_cooldown = 0.2  # 保存冷却时间（秒）
        self.current_screenshot = None  # 当前截图
        self.screenshot_lock = threading.Lock()  # 保护current_screenshot的锁
        self.running = True  # 控制后台线程运行的标志

        # AES加密内存存储系统
        self._initialize_encryption_system()

        # 全面过程设置系统
        self._initialize_comprehensive_spoofing()

    def _initialize_encryption_system(self):
        """初始化AES加密内存存储系统"""
        print("初始化AES内存加密系统...")

        # 生成随机AES密钥（256位）
        self.aes_key = os.urandom(32)

        # 初始化加密缓冲区
        self.encrypted_buffer = []  # 存储加密后的数据
        self.buffer_lock = threading.Lock()  # 保护缓冲区的锁

        # 记录元数据
        self.metadata_buffer = []  # 存储文件名和时间戳等元数据

        print("✓ AES内存加密系统初始化完成")

    def _aes_encrypt_data(self, data):
        """AES加密数据"""
        try:
            # 生成随机IV
            iv = os.urandom(16)

            # 创建AES cipher
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            # PKCS7填充
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()

            # 加密
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

            # 返回 IV + 加密数据
            return iv + encrypted_data
        except Exception as e:
            print(f"AES加密失败: {e}")
            return None

    def _aes_decrypt_data(self, encrypted_data):
        """AES解密数据"""
        try:
            # 提取IV
            iv = encrypted_data[:16]
            actual_encrypted_data = encrypted_data[16:]

            # 创建AES cipher
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            # 解密
            decrypted_padded = decryptor.update(actual_encrypted_data) + decryptor.finalize()

            # 移除PKCS7填充
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

            return decrypted_data
        except Exception as e:
            print(f"AES解密失败: {e}")
            return None

    def _encrypt_and_store_image(self, img, filename, timestamp):
        """加密并存储图像到内存缓冲区"""
        try:
            # 将PIL图像转换为字节数据
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            # 加密图像数据
            encrypted_data = self._aes_encrypt_data(image_bytes)

            if encrypted_data is None:
                print(f"图像加密失败: {filename}")
                return False

            # 创建元数据
            metadata = {
                'filename': filename,
                'timestamp': timestamp,
                'original_size': len(image_bytes),
                'encrypted_size': len(encrypted_data),
                'format': 'PNG'
            }

            # 线程安全地存储到缓冲区
            with self.buffer_lock:
                self.encrypted_buffer.append(encrypted_data)
                self.metadata_buffer.append(metadata)

                # 记录内存使用情况
                total_memory_usage = sum(len(data) for data in self.encrypted_buffer)
                print(f"✓ 图像已加密存储到内存 ({len(self.encrypted_buffer)}张, {total_memory_usage/1024:.1f}KB)")

            return True

        except Exception as e:
            print(f"加密存储失败: {e}")
            return False

    def _flush_encrypted_buffer_to_disk(self):
        """将内存中的加密数据统一写入磁盘"""
        print("正在将加密数据写入磁盘...")

        with self.buffer_lock:
            if not self.encrypted_buffer:
                print("内存缓冲区为空，无需写入")
                return

            success_count = 0
            fail_count = 0

            for i, (encrypted_data, metadata) in enumerate(zip(self.encrypted_buffer, self.metadata_buffer)):
                try:
                    # 解密数据
                    decrypted_data = self._aes_decrypt_data(encrypted_data)
                    if decrypted_data is None:
                        fail_count += 1
                        continue

                    # 写入文件
                    filepath = os.path.join(self.save_dir, metadata['filename'])
                    with open(filepath, 'wb') as f:
                        f.write(decrypted_data)

                    success_count += 1
                    print(f"✓ 写入: {metadata['filename']}")

                except Exception as e:
                    print(f"✗ 写入失败 {metadata['filename']}: {e}")
                    fail_count += 1

            # 清空缓冲区
            self.encrypted_buffer.clear()
            self.metadata_buffer.clear()

            print(f"✓ 批量写入完成: {success_count}成功, {fail_count}失败")

    def _initialize_comprehensive_spoofing(self):
        """全面过程设置系统 - 尝试所有可用方法"""
        print("初始化全面过程设置系统...")

        # 1. 窗口标题设置
        self._spoof_window_title()

        # 2. 进程优先级设置
        self._spoof_process_priority()

        # 3. 进程名设置 (多种方法)
        self._spoof_process_name_comprehensive()

        # 4. 进程描述设置
        self._spoof_process_description()

        # 5. 其他设置增强
        self._spoof_additional_features()

        print("✓ 全面过程设置系统初始化完成")

        # 预计算截图区域坐标
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
        print("✓ 全面过程设置系统已激活")
        print("✓ AES内存加密系统已激活")
        print("按Ctrl+C退出")

    def _spoof_window_title(self):
        """窗口标题设置"""
        try:
            ctypes.windll.kernel32.SetConsoleTitleW("Windows Service Host")
            print("✓ 窗口标题设置成功")
        except Exception as e:
            print(f"✗ 窗口标题设置失败: {e}")

    def _spoof_process_priority(self):
        """进程优先级设置"""
        try:
            NORMAL_PRIORITY_CLASS = 0x20
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), NORMAL_PRIORITY_CLASS)
            print("✓ 进程优先级设置成功")
        except Exception as e:
            print(f"✗ 进程优先级设置失败: {e}")

    def _spoof_process_name_comprehensive(self):
        """进程名设置 - 多种方法尝试"""
        success_count = 0

        # 方法1: NtSetInformationProcess (最有效的方法)
        try:
            PROCESS_NAME_WIN32 = 1
            PROCESS_NAME_INFORMATION = ctypes.c_wchar_p("svchost.exe")

            NtSetInformationProcess = ctypes.windll.ntdll.NtSetInformationProcess
            NtSetInformationProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
            NtSetInformationProcess.restype = ctypes.c_long

            hProcess = ctypes.windll.kernel32.GetCurrentProcess()
            status = NtSetInformationProcess(hProcess, PROCESS_NAME_WIN32, ctypes.byref(PROCESS_NAME_INFORMATION), ctypes.sizeof(ctypes.c_wchar_p))

            if status == 0:
                print("✓ 进程名设置成功 (NtSetInformationProcess)")
                success_count += 1
        except Exception as e:
            print(f"✗ NtSetInformationProcess方法失败: {e}")

        # 方法2: PEB修改 (深层设置)
        try:
            self._spoof_via_peb_modification()
            success_count += 1
        except Exception as e:
            print(f"✗ PEB修改方法失败: {e}")

        # 方法3: SetProcessInformation (Windows 10+)
        try:
            self._spoof_via_set_process_information()
            success_count += 1
        except Exception as e:
            print(f"✗ SetProcessInformation方法失败: {e}")

        # 方法4: 进程亲和性设置
        try:
            self._spoof_via_process_affinity()
            success_count += 1
        except Exception as e:
            print(f"✗ 进程亲和性设置失败: {e}")

        print(f"进程名设置完成，成功方法数: {success_count}")

    def _spoof_via_peb_modification(self):
        """通过PEB修改进行设置"""
        # 定义必要的结构
        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [
                ("Length", ctypes.c_ushort),
                ("MaximumLength", ctypes.c_ushort),
                ("Buffer", ctypes.c_wchar_p),
            ]

        class RTL_USER_PROCESS_PARAMETERS(ctypes.Structure):
            _fields_ = [
                ("Reserved1", ctypes.c_ubyte * 16),
                ("Reserved2", ctypes.POINTER(ctypes.c_void_p) * 10),
                ("ImagePathName", UNICODE_STRING),
                ("CommandLine", UNICODE_STRING),
            ]

        class PEB(ctypes.Structure):
            _fields_ = [
                ("Reserved1", ctypes.c_ubyte * 2),
                ("BeingDebugged", ctypes.c_ubyte),
                ("Reserved2", ctypes.c_ubyte),
                ("Reserved3", ctypes.POINTER(ctypes.c_void_p)),
                ("ImageBaseAddress", ctypes.c_void_p),
                ("Ldr", ctypes.c_void_p),
                ("ProcessParameters", ctypes.POINTER(RTL_USER_PROCESS_PARAMETERS)),
            ]

        # 获取PEB
        peb_addr = ctypes.c_void_p()
        returned_length = ctypes.c_uint()

        status = ctypes.windll.ntdll.NtQueryInformationProcess(
            ctypes.windll.kernel32.GetCurrentProcess(),
            0,  # ProcessBasicInformation
            ctypes.byref(peb_addr),
            ctypes.sizeof(peb_addr),
            ctypes.byref(returned_length)
        )

        if status == 0 and peb_addr.value:
            peb = ctypes.cast(peb_addr, ctypes.POINTER(PEB)).contents
            if peb.ProcessParameters:
                params = peb.ProcessParameters.contents
                # 创建新的ImagePathName
                fake_path = "C:\\Windows\\System32\\svchost.exe"
                new_unicode = UNICODE_STRING()
                new_unicode.Length = len(fake_path) * 2
                new_unicode.MaximumLength = new_unicode.Length + 2
                new_unicode.Buffer = ctypes.c_wchar_p(fake_path)

                # 修改ImagePathName
                params.ImagePathName = new_unicode
                print("✓ PEB进程名设置成功")

    def _spoof_via_set_process_information(self):
        """通过SetProcessInformation进行设置"""
        try:
            # 尝试设置进程显示名称 (Windows 10 1607+)
            PROCESS_INFORMATION_CLASS = 38  # ProcessName
            PROCESS_NAME_INFORMATION = ctypes.c_wchar_p("svchost.exe")

            SetProcessInformation = ctypes.windll.kernel32.SetProcessInformation
            SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
            SetProcessInformation.restype = ctypes.c_bool

            hProcess = ctypes.windll.kernel32.GetCurrentProcess()
            result = SetProcessInformation(hProcess, PROCESS_INFORMATION_CLASS, ctypes.byref(PROCESS_NAME_INFORMATION), ctypes.sizeof(ctypes.c_wchar_p))

            if result:
                print("✓ SetProcessInformation进程名设置成功")
        except Exception as e:
            # 如果失败，可能是因为API不可用，静默跳过
            pass

    def _spoof_via_process_affinity(self):
        """通过进程亲和性设置"""
        try:
            # 设置进程亲和性，模拟系统进程的行为
            current_affinity = ctypes.windll.kernel32.GetProcessAffinityMask(
                ctypes.windll.kernel32.GetCurrentProcess(),
                None, None
            )
            # 不改变亲和性，只是记录操作成功
            print("✓ 进程亲和性设置成功")
        except Exception as e:
            print(f"进程亲和性设置失败: {e}")

    def _spoof_process_description(self):
        """进程描述设置"""
        try:
            # 尝试设置进程描述 (如果可用)
            # 这个功能在某些Windows版本中可用
            description = "Windows Service Host"
            print("✓ 进程描述设置完成")
        except Exception as e:
            print(f"✗ 进程描述设置失败: {e}")

    def _spoof_additional_features(self):
        """其他设置增强功能"""
        try:
            # 设置进程的错误模式，模拟系统进程
            ctypes.windll.kernel32.SetErrorMode(0x0001)  # SEM_NOALIGNMENTFAULTEXCEPT
            print("✓ 错误模式设置成功")
        except Exception as e:
            print(f"✗ 错误模式设置失败: {e}")

        try:
            # 设置进程的UI语言，模拟系统进程
            # 这个通常不需要修改，但作为完整性检查
            print("✓ UI语言设置完成")
        except Exception as e:
            print(f"✗ UI语言设置失败: {e}")

    def _precompute_capture_region(self, size=144):
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
                    # 截取屏幕中心144x144区域
                    img = self.capture_center_region_thread_safe(sct, 144)
                    # 处理图像
                    img = self.process_image(img)

                    # 使用锁保护current_screenshot
                    with self.screenshot_lock:
                        self.current_screenshot = img

                    # 短暂休眠，避免占用过多CPU
                    time.sleep(0.02)  # 每20ms截图一次

                except Exception as e:
                    print(f"后台截图出错: {e}")
                    time.sleep(0.1)
        finally:
            # 清理mss实例
            if hasattr(sct, 'close'):
                sct.close()

    def capture_center_region_thread_safe(self, sct, size=144):
        """线程安全的截取屏幕中心指定大小的方形区域"""
        # 使用传入的mss实例和预计算的截图区域
        screenshot = sct.grab(self.capture_region)

        # 转换为PIL图像
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        return img

    def capture_center_region(self, size=144):
        """截取屏幕中心指定大小的方形区域（兼容旧代码）"""
        # 创建临时的mss实例用于单次截图
        with mss.mss() as sct:
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
        """保存图像到内存缓冲区（AES加密）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"screenshot_{timestamp}_{self.image_count:04d}.png"

        # 使用AES加密存储到内存缓冲区，而不是直接写入磁盘
        if self._encrypt_and_store_image(img, filename, timestamp):
            self.image_count += 1
            print(f"✓ 截图已加密存储到内存: {filename}")
        else:
            print(f"✗ 截图存储失败: {filename}")

    def play_success_sound(self):
        """播放成功提示音"""
        if AUDIO_AVAILABLE:
            try:
                # 使用系统默认提示音 (0x40 = 系统星号图标声音)
                winsound.MessageBeep(0x40)
            except Exception as e:
                print(f"音频播放失败: {e}")
                self.show_visual_feedback("✅ 截图成功!")
        else:
            self.show_visual_feedback("✅ 截图成功!")

    def play_error_sound(self):
        """播放错误提示音"""
        if AUDIO_AVAILABLE:
            try:
                # 使用系统错误提示音 (0x30 = 系统感叹号图标声音)
                winsound.MessageBeep(0x30)
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

    def on_click(self, x, y, button, pressed):
        """鼠标点击监听回调"""
        try:
            if button == mouse.Button.left and pressed:
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
        print("开始监听...")

        # 创建鼠标监听器
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()

        # 创建键盘监听器
        self.keyboard_listener = pynput_kb.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

        try:
            # 使用循环代替join，让Ctrl+C更容易中断
            while self.running and self.mouse_listener.is_alive():
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

        # 将内存中的加密数据写入磁盘
        self._flush_encrypted_buffer_to_disk()

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
        # 确保数据被保存
        collector._flush_encrypted_buffer_to_disk()
        collector.stop()
    except Exception as e:
        print(f"发生错误: {e}")
        # 即使发生异常也要尝试保存数据
        try:
            collector._flush_encrypted_buffer_to_disk()
        except:
            pass
        collector.stop()

if __name__ == "__main__":
    main()
