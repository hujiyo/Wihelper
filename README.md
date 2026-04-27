# WiHelper

基于机器学习的项目，用于屏幕中心特定目标的实时推理与自动响应

## 项目结构

```
WiHelper/
├── wihelper.py              # 主程序 - 实时推理与自动响应
├── screenshot_collector.py  # 数据收集 - 截图采集器
├── select_helper.py         # 数据标注 - GUI标注工具
├── train_model.py           # 模型训练 - CNN训练脚本
├── inference.py             # 模型推理 - 单独测试脚本
├── environment.yml          # Conda环境配置
├── image/                   # 数据集目录
│   ├── train/got/           # 训练集-有目标
│   ├── train/nogot/         # 训练集-无目标
│   ├── test/got/            # 测试集-有目标
│   └── test/nogot/          # 测试集-无目标
└── models/                  # 模型权重输出目录
```
## 快速开始

### 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate wi
```

### 运行主程序

```bash
python wihelper.py
```

主程序启动后：
- **右键点击**：进入瞄准模式（0.5秒瞄准 + 最长4秒检测窗口）
- **检测到目标**：自动模拟行动

## 数据采集流程

### 1. 收集截图

```bash
python screenshot_collector.py
```

- **左键点击**：保存当前屏幕中心144×144区域截图
- **左Alt+左键**：忽略本次左键点击
- **Ctrl+C**：退出并保存所有数据
> 截图会先AES加密存储在内存中，程序退出时统一写入 `image/` 目录。

### 2. 标注数据

```bash
python select_helper.py
```

GUI界面操作：
- **空格键**：标记为无目标 → 移动到 `train/nogot/`
- **回车键**：标记为有目标 → 移动到 `train/got/`
- **Delete键**：删除低质量图片
- **Ctrl+Z**：撤销操作（最多5步）

### 3. 训练模型

```bash
python train_model.py
```
训练特性：
- 学习率预热 + 余弦退火衰减
- 梯度累积（等效batch_size=128）
- 基于AUC的早停机制
- 动态类别权重平衡

输出文件：
- `models/best_model.keras` - 最佳模型
- `models/final_model.keras` - 最终模型
- `models/info.txt` - 完整训练报告
- `models/confusion_matrix.png` - 混淆矩阵图

### 4. 测试推理

```bash
# 单张图片
python inference.py --image test.png

# 批量测试
python inference.py --batch image/test/

# 速度基准测试
python inference.py --benchmark

# 指定模型和阈值
python inference.py --model models/best_model.keras --threshold 0.8 --image test.png
```

## 技术规格

| 项目 | 规格 |
|------|------|
| 数据集尺寸 | 144×144×3 (RGB) |
| 模型输入尺寸 | 120×120×3 (CenterCrop后) |
| 模型格式 | Keras (.keras) |
| 推理框架 | TensorFlow 2.x |
| 检测阈值 | 0.5（可调） |
| 推理延迟 | ~5-10ms/帧 |

### 数据预处理流程

**训练阶段**：
```
144×144原始图片 → CenterCrop(120×120) → 归一化 → 数据增强 → 模型训练
```

**推理阶段**：
```
截图120×120 → 归一化 → 模型推理
```

**关键设计**：
- 数据集保持144×144（人工标注时查看大图更清晰）
- CenterCrop作为预处理步骤，确保模型输入尺寸一致

### 数据增强策略

- 亮度调整 ±20%
- 颜色通道偏移
- 高斯噪声/模糊（30%概率）
- **不使用几何变换**（保持准星位置精确）

### 安全特性

- AES-256-CBC 内存加密存储
- 进程名/窗口标题伪装
- 延迟批量写入磁盘

## 数据集建议

| 类型 | 数量 | 说明 |
|------|------|------|
| 训练集 | 1000-3000张 | - |
| 测试集 | 100-300张 | got/nogot各半 |
| 标注准确率 | >95% | 确保标签正确 |

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 模型文件不存在 | 先运行 `train_model.py` 训练模型 |
| GPU未检测到 | 检查CUDA/cuDNN安装，或使用CPU推理 |
| 截图保存失败 | 确保程序正常退出（Ctrl+C），触发磁盘写入 |
| 推理速度慢 | 首次运行会自动转换为SavedModel格式加速 |
| 误判率高 | 增加训练数据，或调高检测阈值 |
