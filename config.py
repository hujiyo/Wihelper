"""
WiHelper 项目配置文件

集中管理所有硬编码的配置参数（数据、模型、训练、推理、UI、路径、设备）。
主程序 wihelper.py 仍可读取 config.ini 进行用户级覆盖；config.py 提供
代码级默认值与无法在运行时调整的常量。

模块结构:
    DataConfig        - 数据集与图像预处理
    ModelConfig       - 模型架构与 ONNX 导出
    TrainingConfig    - 训练超参
    LossConfig        - 损失与评估相关
    InferenceConfig   - 推理/校验/基准测试
    AppConfig         - 主程序 wihelper.py 运行参数
    ScreenshotConfig  - 截图收集器
    UIConfig          - 数据标注助手 GUI
    PathsConfig       - 通用文件路径
    DeviceConfig      - 设备选择
"""

import os
import sys


# 项目根目录（config.py 与各脚本位于同一目录）
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ==================== 路径配置（先定义，供其他类引用 DATA_ROOT） ====================
class PathsConfig:
    """通用文件路径配置。必须先于 DataConfig/ScreenshotConfig/UIConfig 定义，
    因后者通过 PathsConfig.DATA_ROOT 引用统一的数据根目录。"""

    # 数据根目录：DataConfig.DATA_DIR / ScreenshotConfig.SAVE_DIR /
    # UIConfig.SOURCE_DIR / UIConfig.TARGET_BASE 统一引用此值
    DATA_ROOT = "image"

    MODEL_SAVE_DIR = "models"

    # 模型 / 报告文件命名
    BEST_MODEL_FILENAME = "best_model.pth"
    FINAL_MODEL_FILENAME = "final_model.pth"
    TRAIN_INFO_FILENAME = "info.txt"
    CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

    # best_model.pth 候选路径（按顺序查找）
    BEST_MODEL_CANDIDATES = (
        "models/best_model.pth",
        "models-v1.1-4/best_model.pth",
    )
    # 查找 best_model 时的目录
    BEST_MODEL_DIRS = ("models", "models-v1.1-4")

    # find_mislabeled.py 的源文件夹
    MISLABELED_FOLDERS = {
        "image/train/got": 1,      # 人工标注: 有目标
        "image/train/nogot": 0,    # 人工标注: 无目标
    }
    MISLABELED_OUT_DIR = "image"


# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""
    # 数据集目录：统一引用 PathsConfig.DATA_ROOT
    DATA_DIR = PathsConfig.DATA_ROOT

    # 模型输入尺寸（CenterCrop 后）
    TARGET_SIDE_LENGTH = 120

    # 数据子目录约定
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    GOT_DIR = "got"        # 有目标
    NOGOT_DIR = "nogot"    # 无目标

    # 标签映射: ImageFolder 会按字母排序 got=0, nogot=1，
    # 这里定义为最终语义标签
    LABEL_MAP = {"got": 1, "nogot": 0}

    # 支持的图片扩展名
    SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp")
    SUPPORTED_EXT_FULL = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

    # 文件名启发式推断标签
    TARGET_KEYWORDS = ("target", "got", "hit", "aim")
    NOGOT_KEYWORDS = ("nogot", "miss", "nohit", "notarget")


# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构与 ONNX 导出相关参数"""
    # ----- 架构（WiHelperCNN，120x120 输入，空洞卷积版） -----
    BLOCK1_IN_CHANNELS = 3
    BLOCK1_OUT_CHANNELS = 32
    BLOCK2_IN_CHANNELS = 32
    BLOCK2_OUT_CHANNELS = 64
    # compress 阶段: 108 -> 54 -> 27 -> 14
    COMPRESS_CHANNELS = (64, 32, 32)
    COMPRESS_STRIDE = 2
    # 中心裁剪到 108x108，再三层 stride=2 压缩到 14x14 = 196
    # 最终 feature map 维度 = COMPRESS_CHANNELS[-1] * 14 * 14 = 6272
    CENTER_CROP_SIZE = 108
    CLASSIFIER_HIDDEN = 256
    CLASSIFIER_DROPOUT = 0.3
    NUM_CLASSES = 1

    # ----- ONNX 导出 -----
    INPUT_NAME = "image"
    OUTPUT_NAME = "logit"
    INPUT_SHAPE = (1, 3, 120, 120)
    ONNX_OPSET = 17
    DYNAMIC_AXES = None  # 固定 batch=1，优化推理性能

    # ----- ONNX 侧基准测试（与 PyTorch 侧 InferenceConfig.BENCHMARK_RUNS 区分） -----
    BENCHMARK_WARMUP_ONNX = 10
    BENCHMARK_RUNS_ONNX = 200

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""
    BATCH_SIZE = 32
    BATCHES_PER_EPOCH = 100
    EPOCHS = 50

    # AdamW 学习率调度（warmup + cosine）
    WARMUP_EPOCHS = 5
    INITIAL_LR = 1e-4
    PEAK_LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0 #浅层模型无需权值衰减

    # 梯度累积
    ACCUMULATION_STEPS = 4

    # DataLoader
    NUM_WORKERS = 0
    PIN_MEMORY = True  # 训练时启用；CPU 模式下请设 False

    # 随机种子
    NP_SEED = 42
    TORCH_SEED = 42

    # 在线增强 (add_noise_and_blur)
    AUG_PROBS = (0.3, 0.3, 0.4)  # noise / blur / none
    NOISE_STD = 5
    BLUR_KSIZES = (3, 5)

# ==================== 损失函数配置 ====================
class LossConfig:
    """损失与评估相关配置"""
    # 战术得分: 每次误报扣分权重
    TACTICAL_FP_PENALTY = 5.0

    # 评估/测试阶段的默认分类阈值
    TEST_THRESHOLD = 0.5


# ==================== 推理参数 ====================
class InferenceConfig:
    """推理/校验/基准测试相关参数（PyTorch 训练/评估侧）"""

    # 训练/评估脚本使用的 .pth 模型路径
    DEFAULT_PTH_MODEL_PATH = "models/best_model.pth"
    # 训练/评估的默认分类阈值（评估/标注用 0.5，开火见 AppConfig.DEFAULT_FIRE_THRESHOLD）
    DEFAULT_TRAIN_THRESHOLD = 0.5

    # ONNX 一致性验证
    VERIFY_NUM_SAMPLES = 100
    VERIFY_TOLERANCE = 1e-4
    # 误差小于该值视为"实际可接受"
    VERIFY_ACCEPTABLE_DIFF = 1e-3

    # 基准测试（PyTorch 侧；ONNX 侧基准见 ModelConfig.BENCHMARK_RUNS_ONNX）
    BENCHMARK_WARMUP = 10
    BENCHMARK_RUNS = 100

    # find_mislabeled: 概率落在 [UNCERTAIN_LOW, UNCERTAIN_HIGH] 之间视为不确定样本
    UNCERTAIN_LOW = 0.2
    UNCERTAIN_HIGH = 0.8


# ==================== 主程序运行参数 ====================
class AppConfig:
    """主程序 wihelper.py 的运行参数（ONNX Runtime，无 PyTorch 依赖）"""
    # 用户可在 config.ini 覆盖的项
    # 开火阈值：保守值（>0.5）以减少误开火
    DEFAULT_FIRE_THRESHOLD = 0.8
    DEFAULT_FIRE_COOLDOWN = 4.0
    # 主程序加载的 ONNX 模型路径（与 InferenceConfig.DEFAULT_PTH_MODEL_PATH 区分）
    DEFAULT_ONNX_MODEL_PATH = "wihelper_model.onnx"
    DEFAULT_TARGET_FPS = 60

    # 通用系统音效（供主程序与截图工具共用，集中放这里避免分散在 ScreenshotConfig）
    SOUND_SUCCESS = 0x40
    SOUND_ERROR = 0x30
    # 控制台伪装的进程名/窗口标题
    CONSOLE_TITLE = "Windows Service Host"

    # FPS 限幅
    MIN_FPS = 1
    MAX_FPS = 240

    # 屏幕中心截图尺寸（与 ModelConfig.INPUT_SHAPE 保持一致）
    CAPTURE_SIZE = 120 #这个是Wihelper等脚本使用的截取大小，区别于数据收集时的144

    # 瞄准模式
    AIMING_TIME = 0.5       # 起始 0.5s 纯瞄准，不开火
    TOTAL_TIMEOUT = 4.0      # 整体超时（aiming + firing）
    MAX_FIRE_COUNT = 8       # 连狙模式 4s 内最多 8 枪

    # 模式判定: fire_cooldown >= SNIPER_COOLDOWN_THRESHOLD 视为大狙
    SNIPER_COOLDOWN_THRESHOLD = 4.0

    # 按键
    FIRE_KEY = "p"           # 开火键
    INTERRUPT_KEY = "f"      # 打断判断模式

    # 推理模块
    WARMUP_RUNS = 5
    WARMUP_DUMMY_CHANNELS = 4  # BGRA

    # 主循环
    MEMORY_CHECK_INTERVAL = 1000
    DEBUG_INTERVAL = 0.1       # 调试日志刷新间隔（秒）

    # 资源清理
    MOUSE_LISTENER_JOIN_TIMEOUT = 1.0
    THREAD_JOIN_TIMEOUT = 1.0

    # FPS 统计
    FPS_STAT_INTERVAL = 5.0  # 每隔 N 秒打印一次平均 FPS

    # config.ini 段名
    CONFIG_INI_SECTION = "wihelper"
    CONFIG_INI_FILENAME = "config.ini"


# ==================== 截图收集器参数 ====================
class ScreenshotConfig:
    """截图收集器 screenshot_collector.py 相关参数"""

    # 保存目录：统一引用 PathsConfig.DATA_ROOT
    SAVE_DIR = PathsConfig.DATA_ROOT
    SAVE_COOLDOWN = 0.2          # 秒
    BACKGROUND_CAPTURE_INTERVAL = 0.02      # 后台截图间隔（秒）

    # 屏幕中心截图区域
    RAW_CAPTURE_SIZE = 144       # 后台持续截取的方形边长（大图人工分类更清晰，小图用于训练）

    # 截图文件名
    FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
    FILENAME_PATTERN = "screenshot_{timestamp}_{index:04d}.png"


# ==================== 数据标注助手 UI 参数 ====================
class UIConfig:
    """数据标注助手 select_helper.py 的 GUI 参数"""

    # 源/目标目录：统一引用 PathsConfig.DATA_ROOT
    SOURCE_DIR = PathsConfig.DATA_ROOT
    TARGET_BASE = PathsConfig.DATA_ROOT
    MAX_UNDO = 5

    WINDOW_SIZE = "520x700"
    WINDOW_PADDING = 10        # ttk.Frame 接受 int，无需 str() 转换
    WINDOW_MIN_WIDTH = 100
    WINDOW_MIN_HEIGHT = 200
    SHOW_FIRST_IMAGE_DELAY_MS = 300

    # 撤销时回收目录
    RECYCLE_DIR = ".recycle"

    # 标签颜色（按模型预测置信度）
    PRED_COLORS = {
        # 预测有目标（红色系）
        "got_high":     "#CC0000",  # >=0.8
        "got_medium":   "#DD6644",  # >=0.6
        "got_low":      "#DDAA44",  # <0.6
        # 预测无目标（蓝色系）
        "nogot_high":   "#0066CC",  # <=0.2
        "nogot_medium": "#4488CC",  # <=0.4
        "nogot_low":    "#88AACC",  # <0.4
    }
    PRED_TEXT_COLOR = "white"
    PRED_BG_DEFAULT = "gray"
    PRED_FONT = ("Arial", 18, "bold")
    PRED_BG_HIGH_GOT_THRESHOLD = 0.8
    PRED_BG_MEDIUM_GOT_THRESHOLD = 0.6
    PRED_BG_HIGH_NOGOT_THRESHOLD = 0.2
    PRED_BG_MEDIUM_NOGOT_THRESHOLD = 0.4

# ==================== 设备配置 ====================
class DeviceConfig:
    """设备选择。训练脚本要求 CUDA；主程序 wihelper.py 允许 CPU 兜底。"""
    @staticmethod
    def get_device(require_cuda: bool = False):
        """
        获取 PyTorch 设备。

        延迟导入 torch，避免在 ONNX-only 运行路径（wihelper.py、
        screenshot_collector.py 等仅做 ONNX 推理/截图的脚本）下污染
        运行环境。

        Args:
            require_cuda: 若为 True，则在 CUDA 不可用时打印错误并 sys.exit(1)；
                          若为 False，则在 CUDA 不可用时退回 CPU。
        """
        import torch  # 延迟导入，避免污染纯 ONNX 运行路径

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            return device
        if require_cuda:
            print("ERROR:CUDA 不可用，程序退出")
            sys.exit(1)
        return device
