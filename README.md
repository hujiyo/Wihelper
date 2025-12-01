# 🎯 WiHelper

基于深度学习的屏幕中心目标识别工具，支持实时推理与自动响应。

## 📁 项目结构

```
WiHelper/
├── wihelper.py              # 主程序 - 实时识别与自动响应
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
└── models/                  # 模型文件
    ├── best_model.h5        # 最佳模型
    └── best_model_savedmodel/  # SavedModel格式（自动生成）
```

## 🚀 快速开始

### 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate wihelper
```

### 运行主程序

```bash
python wihelper.py
```

主程序启动后：
- **右键点击**：进入瞄准模式（0.5秒瞄准 + 最长4秒检测窗口）
- **检测到目标**：自动模拟按下 `P` 键
- **双重检测机制**：连续两帧确认后才触发，减少误判
- **Ctrl+C**：退出程序

## 📸 数据采集流程

### 1. 收集截图

```bash
python screenshot_collector.py
```

- **左键点击**：保存当前屏幕中心144×144区域截图
- **左Alt+左键**：忽略本次点击（防误触）
- **Ctrl+C**：退出并保存所有数据

截图会先AES加密存储在内存中，程序退出时统一写入 `image/` 目录。

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

训练时可选择三种模型架构：
1. **平衡微调版**（默认）- 3 Block + Flatten，约2万特征
2. **极简轻量版** - 4 Block + GAP，64维特征
3. **加强版GAP** - 4 Block宽通道 + GAP，256维特征

训练特性：
- 学习率预热 + 余弦退火衰减
- 梯度累积（等效batch_size=128）
- 基于AUC的早停机制
- 动态类别权重平衡

### 后训练机制

基础训练完成后，程序会询问是否进行后训练。后训练专门针对**难分类样本**（预测置信度在0.2-0.8之间）进行强化学习：

- 每轮自动筛选测试集中的难分类样本
- 使用恒定学习率 1e-4
- 基于训练损失判断最佳模型
- 每轮结束后可选择继续或停止
- 难分类样本归零时自动结束

输出文件：
- `models/best_model.h5` - 最佳模型
- `models/final_model.h5` - 最终模型
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
python inference.py --model models/best_model.h5 --threshold 0.8 --image test.png
```

## ⚙️ 技术规格

| 项目 | 规格 |
|------|------|
| 输入尺寸 | 144×144×3 (RGB) |
| 模型格式 | H5 / SavedModel |
| 推理框架 | TensorFlow 2.x |
| 检测阈值 | 0.5（可调） |
| 推理延迟 | ~5-10ms/帧 |

### 数据增强策略

- 亮度调整 ±20%
- 颜色通道偏移
- 高斯噪声/模糊（30%概率）
- **不使用几何变换**（保持准星位置精确）

### 安全特性

- AES-256-CBC 内存加密存储
- 进程名/窗口标题伪装
- 延迟批量写入磁盘

## 📊 数据集建议

| 类型 | 数量 | 说明 |
|------|------|------|
| 训练集 | 400-1000张 | got/nogot各半 |
| 测试集 | 100-200张 | got/nogot各半 |
| 标注准确率 | >95% | 确保标签正确 |

## 🔧 常见问题

| 问题 | 解决方案 |
|------|----------|
| 模型文件不存在 | 先运行 `train_model.py` 训练模型 |
| GPU未检测到 | 检查CUDA/cuDNN安装，或使用CPU推理 |
| 截图保存失败 | 确保程序正常退出（Ctrl+C），触发磁盘写入 |
| 推理速度慢 | 首次运行会自动转换为SavedModel格式加速 |
| 误判率高 | 增加训练数据，或调高检测阈值 |

## 📝 更新日志

- 主程序支持大狙/连狙双模式（通过 `fire_cooldown` 参数切换）
- 双重检测机制减少误触发
- SavedModel自动转换加速推理
- 后训练功能针对难分类样本优化