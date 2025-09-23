# 🎯 WiHelper

## 📋 项目概述

WiHelper项目是一个简单的基于深度学习的游戏人物识别工具，它通过图像识别技术判断是否发现目标并通过提供的API发出相应的信息。项目包含了一整套从训练数据收集到模型训练和推理的整套流程框架可供定制。注意，WiHelper只对外提供API接口以输出响应信号，并不直接涉及具体的响应处理后的行为方式。

## 🏗️ 系统架构

项目的系统架构遵循一个标准化的机器学习工作流，包括：
`数据收集 → 数据标注 → 模型训练 → 模型推理 → 提供信号`。

## 📁 项目文件结构

```
WiHelper/
├── 📊 数据处理
│   ├── screenshot_collector.py    # 屏幕截图数据收集器
│   └── select_helper.py          # 数据标注辅助工具
│
├── 🤖 模型训练
│   ├── train_model.py            # CNN模型训练脚本
│   ├── inference.py              # 模型推理脚本
│   └── models/                   # 训练好的模型文件
│
├── 📦 配置与依赖
│   ├── requirements.txt           # Python依赖包
│   ├── activate_env.bat           # 环境激活脚本
│   └── image/                     # 数据集目录
│       ├── train/                 # 训练数据
│       └── test/                  # 测试数据
│
└──── README.md                  # 项目说明
```

## 🚀 使用流程

### 阶段零：环境配置 ⚙️

#### **依赖包更新**
项目已添加AES内存加密功能，需要安装：
- `cryptography>=42.0.0` - 用于AES加密/解密

#### **环境激活**
```bash
# 使用自动激活脚本
activate_env.bat

# 或手动激活
conda activate wihelper
```

1. **激活conda环境**：

    ```bash
    # 使用自动激活脚本（推荐）
    activate_env.bat

    # 或手动激活
    conda activate wihelper
    ```

2. **验证环境**：

    ```bash
    activate_env.bat test
    ```

### 阶段一：数据收集 📸

1.  **AES内存加密存储功能**：截图数据现在会先用AES加密存储在内存中，程序退出时统一写入磁盘，避免实时文件I/O检测。

2.  激活环境并运行脚本：

    ```bash
    # 方法1：使用激活脚本
    activate_env.bat run screenshot_collector.py

    # 方法2：手动激活后运行
    conda activate wihelper
    python screenshot_collector.py
    ```

默认按鼠标右键即可截图，图片将自动保存到 `image/` 目录中。程序可在后台运行，无需保持程序窗口在前台。左ALT键可暂时取消截图。

### 阶段二：数据标注 🏷️

1.  激活环境并运行标注辅助工具：

    ```bash
    # 方法1：使用激活脚本
    activate_env.bat run select_helper.py

    # 方法2：手动激活后运行
    conda activate wihelper
    python select_helper.py
    ```

2.  使用键盘进行操作：

    * **空格键**：标记为"无目标" (`notarget`)。
    * **回车键**：标记为"有目标" (`target`)。

### 阶段三：模型训练 🎓

1.  **准备数据**：确保你已经使用 `select_helper.py` 标注了足够的数据。

```
WiHelper/
├── image/
│   ├── train/
│   │   ├── got/     # 有目标的训练图片
│   │   └── nogot/   # 无目标的训练图片
│   └── test/
│       ├── got/     # 有目标的测试图片
│       └── nogot/   # 无目标的测试图片
└── models/
```

2.  **开始训练**：

    ```bash
    # 方法1：使用激活脚本
    activate_env.bat run train_model.py

    # 方法2：手动激活后运行
    conda activate wihelper
    python train_model.py
    ```

训练完成后，会在 `models/` 目录下生成 `best_model.h5`、`training_history.png`、`confusion_matrix.png` 和 `model_info.json` 等文件。

### 阶段四：模型推理 🔍

  * **单张图片测试**：

    ```bash
    # 方法1：使用激活脚本
    activate_env.bat run inference.py --image test.png

    # 方法2：手动激活后运行
    conda activate wihelper
    python inference.py --image test.png
    ```

  * **批量预测测试**：

    ```bash
    activate_env.bat run inference.py --batch image/test/
    ```

  * **推理速度测试**：

    ```bash
    activate_env.bat run inference.py --benchmark
    ```


## ⚙️ 技术规格

### AES内存加密系统

* **加密算法**：AES-256-CBC
* **密钥管理**：程序运行时动态生成随机密钥
* **存储方式**：内存缓冲区 + 延迟写入
* **安全特性**：
  - 随机IV（初始化向量）
  - PKCS7填充
  - 线程安全的缓冲区访问
  - 程序退出时自动解密写入

### 模型配置

  * **输入尺寸**：144×144×3 (RGB)。
  * **架构**：轻量级CNN，包含3个卷积块和2个全连接层。
  * **参数量**：约 2.1M。
  * **输出**：二分类概率 (0-1)。

### 训练配置

  * **优化器**：Adam。
  * **损失函数**：Binary Crossentropy。
  * **数据增强**：支持缩放、水平翻转等增强方式。
  * **批次大小**：32。
  * **训练轮次**：50 (使用早停机制)。

### 性能目标

  * **目标准确率**：95%+。
  * **推理速度**：10ms/张 (CPU)。
  * **模型大小**：约 \~8MB。

## 📊 数据集要求

### 推荐数据量

  * **训练集**：400-1000张 (各200-500张)。
  * **测试集**：100-200张 (各50-100张)。
  * **总计**：500-1200张图片。

### 数据质量

  * **分辨率**：144×144像素。
  * **格式**：PNG (RGB，3通道)。
  * **标注准确率**：95%+。
  * **类别平衡**：有目标/无目标 ≈ 1:1。

## 🔧 故障排除

  * **AES加密相关问题**：
    - 加密失败：检查cryptography库是否正确安装
    - 解密失败：确保程序正常退出以保存数据
    - 内存使用过高：监控内存缓冲区大小，可考虑定期清理

  * **训练数据不足**：增加数据收集时间，或使用数据增强技术。
  * **模型过拟合**：增加 Dropout 比例、使用早停机制或收集更多样化的数据。
  * **推理速度慢**：可考虑使用 GPU 训练、优化模型架构或减小模型复杂度。
  * **音频反馈不可用**：如果 `winsound` 音频功能不可用，程序会自动切换到视觉反馈。
  * **内存不足**：减小批次大小或图片尺寸，注意AES加密会增加内存使用。

## 🚀 未来扩展与改进

  * **技术改进**：优化模型的结构设计，减少时延。优化截屏技术，压低风险和性能影响
  * **外设功能**: 尝试硬件级的识别技术或更人性化的参数调节机制
  