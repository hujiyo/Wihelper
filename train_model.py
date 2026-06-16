#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#模型训练脚本 - 训练CNN模型来识别是否有目标

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

from config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    PathsConfig,
    TrainingConfig,
    DeviceConfig,
)


class WiHelperCNN(nn.Module):
    """空洞卷积版 - 全程保持空间分辨率，中心裁剪后压缩"""
    def __init__(self):
        super().__init__()
        # 空洞卷积特征提取: 120×120 全程不变
        # Block 1: dilation=1, 感受野 3→5
        self.block1 = nn.Sequential(
            nn.Conv2d(ModelConfig.BLOCK1_IN_CHANNELS, ModelConfig.BLOCK1_OUT_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(ModelConfig.BLOCK1_OUT_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(ModelConfig.BLOCK1_OUT_CHANNELS, ModelConfig.BLOCK1_OUT_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(ModelConfig.BLOCK1_OUT_CHANNELS),
            nn.ReLU(inplace=True),
        )
        # Block 2: dilation=2, 感受野 5→9→13
        self.block2 = nn.Sequential(
            nn.Conv2d(ModelConfig.BLOCK2_IN_CHANNELS, ModelConfig.BLOCK2_OUT_CHANNELS, 3, padding=2, dilation=2),
            nn.BatchNorm2d(ModelConfig.BLOCK2_OUT_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(ModelConfig.BLOCK2_OUT_CHANNELS, ModelConfig.BLOCK2_OUT_CHANNELS, 3, padding=2, dilation=2),
            nn.BatchNorm2d(ModelConfig.BLOCK2_OUT_CHANNELS),
            nn.ReLU(inplace=True),
        )
        # Block 3 已删除: RF=13 足够中心任务, 中心裁剪扩大到 108×108 用满干净区域
        # 中心108×108 → 压缩到 14×14
        c0, c1, c2 = ModelConfig.COMPRESS_CHANNELS
        s = ModelConfig.COMPRESS_STRIDE
        self.compress = nn.Sequential(
            nn.Conv2d(ModelConfig.BLOCK2_OUT_CHANNELS, c0, 3, stride=s, padding=1),   # 108→54
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c1, 3, stride=s, padding=1),   # 54→27
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=s, padding=1),   # 27→14
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        # 分类头 14×14×32 = 6272
        # 中心 108×108 经三次 stride=2 压缩: ceil(108/8) = 14
        feat_h = math.ceil(ModelConfig.CENTER_CROP_SIZE / (s ** 3))
        flatten_dim = c2 * feat_h * feat_h
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, ModelConfig.CLASSIFIER_HIDDEN),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(ModelConfig.CLASSIFIER_HIDDEN),
            nn.Dropout(ModelConfig.CLASSIFIER_DROPOUT),
            nn.Linear(ModelConfig.CLASSIFIER_HIDDEN, ModelConfig.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.block1(x)   # [B, 32, 120, 120]
        x = self.block2(x)   # [B, 64, 120, 120]
        # 中心裁剪108×108: 去掉外圈6像素(RF=13, 半径6), 感受野完全在原图内
        h = (x.shape[2] - ModelConfig.CENTER_CROP_SIZE) // 2
        w = (x.shape[3] - ModelConfig.CENTER_CROP_SIZE) // 2
        x = x[:, :, h:h+ModelConfig.CENTER_CROP_SIZE, w:w+ModelConfig.CENTER_CROP_SIZE]  # [B, 64, 108, 108]
        x = self.compress(x)    # [B, 32, 14, 14]
        x = self.classifier(x)  # [B, 1]
        return x


def center_crop(img, size=DataConfig.TARGET_SIDE_LENGTH):
    """中心裁剪 numpy 图像"""
    h, w = img.shape[:2]
    ch = (h - size) // 2
    cw = (w - size) // 2
    return img[ch:ch + size, cw:cw + size]


def preprocess(path):
    """图像预处理: OpenCV读取 → 中心裁剪144→120 → RGB → /255 → CHW tensor"""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = center_crop(img, DataConfig.TARGET_SIDE_LENGTH)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def find_best_model():
    """自动查找 best 模型文件"""
    for p in PathsConfig.BEST_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    for d in PathsConfig.BEST_MODEL_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "best" in f.lower() and f.endswith(".pth"):
                    return os.path.join(d, f)
    return None


def tactical_score(
    y_true,
    y_pred_prob,
    threshold=LossConfig.TEST_THRESHOLD,
    fp_penalty=LossConfig.TACTICAL_FP_PENALTY,
):
    """
    战术得分: 每次正确检出 +1 分, 每次误报扣 fp_penalty 分
    归一化: score = (TP - fp_penalty * FP) / 实际目标总数
    完美模型 = 1.0, 误报过多会得负分
    """
    y_pred = (y_pred_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_pos = tp + fn
    if total_pos == 0:
        return 0.0
    return (tp - fp_penalty * fp) / total_pos


def print_training_progress(epoch, total_epochs, logs, test_loader=None, model=None, device=None):
    """简单的训练进度显示函数"""
    print(f"\nEpoch {epoch + 1}/{total_epochs} - "
          f"损失: {logs.get('loss', 0):.4f} - 准确率: {logs.get('accuracy', 0):.4f}")

    if test_loader is not None and model is not None:
        print("评估测试集...")
        model.eval()
        all_probs = []
        all_labels = []
        test_loss_total = 0.0
        test_correct = 0
        test_total = 0
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        test_threshold = LossConfig.TEST_THRESHOLD

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                test_loss_total += loss.sum().item()
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                preds = (probs > test_threshold).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        y_true = np.array(all_labels)
        y_pred_prob = np.array(all_probs)
        y_pred = (y_pred_prob > test_threshold).astype(int)

        test_loss = test_loss_total / test_total
        test_accuracy = test_correct / test_total
        try:
            test_auc = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            test_auc = 0.0
        test_tac = tactical_score(y_true, y_pred_prob, threshold=test_threshold)

        print(f"测试损失: {test_loss:.4f} - 测试准确率: {test_accuracy:.4f} - AUC: {test_auc:.4f} - 战术得分: {test_tac:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"混淆矩阵:  预测无目标  预测有目标")
        print(f"实际无目标    {tn:4d}        {fp:4d}")
        print(f"实际有目标    {fn:4d}        {tp:4d}")

        print("\n随机5个样本的预测详情:")
        np.random.seed(epoch)
        sample_indices = np.random.choice(len(y_true), min(5, len(y_true)), replace=False)
        print("样本ID\t真实标签\t预测概率\t预测类别")
        print("-" * 45)
        for idx in sample_indices:
            true_label = int(y_true[idx])
            pred_prob = y_pred_prob[idx]
            pred_class = 1 if pred_prob > test_threshold else 0
            status = "✓" if true_label == pred_class else "✗"
            print(f"{idx:6d}\t{true_label}\t\t{pred_prob:.4f}\t\t{pred_class}\t\t{status}")
        print("-" * 50)

        return test_loss, test_accuracy, test_auc, test_tac
    return None


class CenterCrop:
    """中心裁剪 144→120"""
    def __init__(self, crop_size=DataConfig.TARGET_SIDE_LENGTH):
        self.crop_size = crop_size

    def __call__(self, img):
        # img 是 numpy array (H, W, C) 来自 cv2 读取
        h, w = img.shape[:2]
        crop_h = (h - self.crop_size) // 2
        crop_w = (w - self.crop_size) // 2
        return img[crop_h:crop_h + self.crop_size, crop_w:crop_w + self.crop_size]


class TrainDataset(datasets.ImageFolder):
    """训练数据集 - 使用 OpenCV 读取以支持 CenterCrop 后再增强"""
    # ImageFolder 按字母排序: got=0, nogot=1
    # 但我们需要 nogot=0, got=1 (与原始 TF 代码一致: got=1 表示有目标)
    LABEL_MAP = DataConfig.LABEL_MAP

    def __init__(self, root, transform_pil, target_size=(DataConfig.TARGET_SIDE_LENGTH, DataConfig.TARGET_SIDE_LENGTH)):
        super().__init__(root)
        self.transform_pil = transform_pil
        self.target_size = target_size
        self.center_crop = CenterCrop(DataConfig.TARGET_SIDE_LENGTH)

    def __getitem__(self, index):
        path, label = self.samples[index]
        # 用 OpenCV 读取以保持与原始流程一致
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.center_crop(img)
        from PIL import Image as PILImage
        img = PILImage.fromarray(img)
        if self.transform_pil:
            img = self.transform_pil(img)
        # 修正标签: 0→1(got), 1→0(nogot)
        return img, 1 - label


class WiHelperTrainer:
    def __init__(self, data_dir=DataConfig.DATA_DIR, model_save_dir=PathsConfig.MODEL_SAVE_DIR):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.img_height = DataConfig.TARGET_SIDE_LENGTH
        self.img_width = DataConfig.TARGET_SIDE_LENGTH
        self.batch_size = TrainingConfig.BATCH_SIZE
        self.batches_per_epoch = TrainingConfig.BATCHES_PER_EPOCH
        self.epochs = TrainingConfig.EPOCHS

        os.makedirs(model_save_dir, exist_ok=True)

        # 设备（训练要求 CUDA）
        self.device = DeviceConfig.get_device(require_cuda=True)

        np.random.seed(TrainingConfig.NP_SEED)
        torch.manual_seed(TrainingConfig.TORCH_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(TrainingConfig.TORCH_SEED)

    def create_model(self):
        print("\n" + "=" * 40)
        print("使用 CNN轻量极速版 (120×120输入)")
        print("=" * 40)
        model = WiHelperCNN().to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        return model

    def add_noise_and_blur(self, image):
        enhancement_type = np.random.choice(
            ['noise', 'blur', 'none'],
            p=TrainingConfig.AUG_PROBS,
        )
        if enhancement_type == 'noise':
            noise = np.random.normal(0, TrainingConfig.NOISE_STD, image.shape).astype(np.uint8)
            image = cv2.add(image.astype(np.uint8), noise)
        elif enhancement_type == 'blur':
            ksize = np.random.choice(TrainingConfig.BLUR_KSIZES)
            image = cv2.GaussianBlur(image.astype(np.uint8), (ksize, ksize), 0)
        return image.astype(np.float32)

    def compute_class_weights(self):
        train_dir = os.path.join(self.data_dir, DataConfig.TRAIN_DIR)
        nogot_count = len([f for f in os.listdir(os.path.join(train_dir, DataConfig.NOGOT_DIR))
                          if f.lower().endswith(DataConfig.SUPPORTED_EXT)])
        got_count = len([f for f in os.listdir(os.path.join(train_dir, DataConfig.GOT_DIR))
                        if f.lower().endswith(DataConfig.SUPPORTED_EXT)])
        total_samples = nogot_count + got_count

        if total_samples == 0:
            print("警告: 训练数据为空，无法计算类别权重")
            return {0: 1.0, 1: 1.0}

        nogot_weight = total_samples / (2.0 * nogot_count)
        got_weight = total_samples / (2.0 * got_count)

        class_weights = {0: nogot_weight, 1: got_weight}
        print("📊 类别权重计算:")
        print(f"  - nogot类别: {nogot_count}个样本, 权重: {nogot_weight:.4f}")
        print(f"  - got类别: {got_count}个样本, 权重: {got_weight:.4f}")
        print(f"  - 权重比值: {got_weight / nogot_weight:.4f}")
        return class_weights

    def create_data_loaders(self):
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 自动 /255 并转为 CHW
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = TrainDataset(
            os.path.join(self.data_dir, DataConfig.TRAIN_DIR),
            transform_pil=train_transform,
            target_size=(self.img_height, self.img_width),
        )
        test_dataset = TrainDataset(
            os.path.join(self.data_dir, DataConfig.TEST_DIR),
            transform_pil=test_transform,
            target_size=(self.img_height, self.img_width),
        )

        # 构建平衡采样器: 正负样本等概率被抽中
        targets = [sample[1] for sample in train_dataset.samples]  # ImageFolder 原始标签
        targets = [1 - t for t in targets]  # 翻转为 nogot=0, got=1
        class_counts = [targets.count(0), targets.count(1)]
        sample_weights = [1.0 / class_counts[t] for t in targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.batch_size * self.batches_per_epoch,
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=TrainingConfig.NUM_WORKERS,
            pin_memory=TrainingConfig.PIN_MEMORY and torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=TrainingConfig.NUM_WORKERS,
            pin_memory=TrainingConfig.PIN_MEMORY and torch.cuda.is_available(),
        )

        # 注意: ImageFolder 原始映射是 {'got':0, 'nogot':1}
        # TrainDataset.__getitem__ 已翻转为 nogot=0, got=1
        print(f"类别映射: nogot=0(无目标), got=1(有目标)")
        return train_loader, test_loader

    def save_complete_info(self, model, history, accuracy, class_report):
        info_path = os.path.join(self.model_save_dir, PathsConfig.TRAIN_INFO_FILENAME)
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("基本信息:\n")
            f.write(f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最终测试准确率: {accuracy:.4f}\n")
            f.write(f"实际训练轮次: {len(history['accuracy'])}\n\n")
            f.write(f"图像大小: {self.img_height}x{self.img_width}\n")
            f.write(f"批次大小: {self.batch_size}\n")
            f.write(f"最大轮次: {self.epochs}\n")
            f.write(f"优化器: adamw (initial_lr=1e-4)\n")
            f.write(f"损失函数: BCEWithLogitsLoss + WeightedRandomSampler(1:1)\n")
            f.write(f"指标: accuracy, auc, 战术得分(FP惩罚=5)\n\n")
            f.write("类别权重信息 (仅供参考，未在Loss中叠加):\n")
            f.write(f"nogot类别权重: {self.class_weights[0]:.4f}\n")
            f.write(f"got类别权重: {self.class_weights[1]:.4f}\n")
            f.write(f"权重比值 (got/nogot): {self.class_weights[1]/self.class_weights[0]:.4f}\n\n")
            f.write("-" * 30 + "\n")
            f.write(str(model) + "\n")
            total_params = sum(p.numel() for p in model.parameters())
            f.write(f"\n总参数量: {total_params:,}\n\n")

            f.write("训练历史:\n")
            f.write(f"总训练轮次: {len(history['accuracy'])}\n\n")
            f.write("轮次\t训练准确率\t测试准确率\t训练损失\t测试损失\t训练AUC\t测试AUC\t战术得分\n")
            f.write("-" * 90 + "\n")
            for epoch in range(len(history['accuracy'])):
                f.write(f"{epoch+1:2d}\t")
                f.write(f"{history['accuracy'][epoch]:.4f}\t")
                f.write(f"{history['test_accuracy'][epoch]:.4f}\t")
                f.write(f"{history['loss'][epoch]:.4f}\t")
                f.write(f"{history['test_loss'][epoch]:.4f}\t")
                f.write(f"{history['auc'][epoch]:.4f}\t")
                f.write(f"{history['test_auc'][epoch]:.4f}\t")
                f.write(f"{history['test_tac'][epoch]:.4f}\t" if epoch < len(history['test_tac']) else "\t")
                f.write("\n")

            f.write("\n" + "=" * 50 + "\n最终结果总结:\n" + "=" * 50 + "\n")
            f.write(f"最终训练准确率: {history['accuracy'][-1]:.4f}\n")
            f.write(f"最终测试准确率: {history['test_accuracy'][-1]:.4f}\n")
            f.write(f"最终训练损失: {history['loss'][-1]:.4f}\n")
            f.write(f"最终测试损失: {history['test_loss'][-1]:.4f}\n")
            f.write(f"最终训练AUC: {history['auc'][-1]:.4f}\n")
            f.write(f"最终测试AUC: {history['test_auc'][-1]:.4f}\n")
            if history['test_tac']:
                f.write(f"最终测试战术得分: {history['test_tac'][-1]:.4f}\n")
                best_tac_idx = max(range(len(history['test_tac'])), key=lambda i: history['test_tac'][i])
                f.write(f"最佳战术得分: {history['test_tac'][best_tac_idx]:.4f} (第{best_tac_idx+1}轮)\n")
            f.write("\n")

            f.write("分类报告:\n")
            f.write("-" * 30 + "\n")
            f.write(class_report)
            f.write("\n")

    def evaluate_model(self, model, test_loader):
        print("📊 模型评估")
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images).squeeze(1)
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_pred_prob = np.array(all_probs)
        y_pred = (y_pred_prob > LossConfig.TEST_THRESHOLD).astype(int)

        class_names = ['无目标', '有目标']
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("分类报告:" + report)
        cm = confusion_matrix(y_true, y_pred)
        print("\n混淆矩阵:\n" + str(cm))
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print(f"准确率: {accuracy:.4f}")

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(self.model_save_dir, PathsConfig.CONFUSION_MATRIX_FILENAME), dpi=300, bbox_inches='tight')
        plt.close()
        return accuracy, report

    def train(self):
        print("目标识别模型训练")
        print(f"训练配置: 图像大小 {self.img_height}x{self.img_width}, 批次大小 {self.batch_size}, 轮次 {self.epochs}")

        if not os.path.exists(self.data_dir):
            print("错误: 数据目录不存在")
            print(f"路径: {self.data_dir}")
            return

        print("\n步骤1/3: 创建模型...")
        model = self.create_model()
        print(model)

        return self._train(model)

    def _train(self, model):
        """CNN模型训练流程 - 使用动态类别权重"""
        print("\n步骤2/3: 加载训练数据...")
        train_loader, test_loader = self.create_data_loaders()
        print("数据统计:")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(test_loader.dataset)}")

        self.class_weights = self.compute_class_weights()

        print("\n步骤3/3: 开始训练模型...")
        print(f"  - 最大轮次: {self.epochs}")
        print(f"  - 批次大小: {self.batch_size}")
        print("  - 优化器: AdamW")
        print("  - 损失函数: BCEWithLogitsLoss + 动态类别权重")
        print("  - 指标: Accuracy, AUC")
        print("\n开始训练 (每轮显示进度和测试评估)...")

        # AdamW 解耦式 weight decay, BN 参数和 bias 不加
        warmup_epochs = TrainingConfig.WARMUP_EPOCHS
        initial_lr = TrainingConfig.INITIAL_LR
        peak_lr = TrainingConfig.PEAK_LR
        min_lr = TrainingConfig.MIN_LR
        accumulation_steps = TrainingConfig.ACCUMULATION_STEPS
        weight_decay = TrainingConfig.WEIGHT_DECAY

        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=initial_lr,
        )
        print(f"  - 优化器: AdamW (lr={initial_lr}, weight_decay={weight_decay}, BN/bias 不加 wd)")

        history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'test_loss': [], 'test_accuracy': [], 'test_auc': [], 'test_tac': []
        }
        best_state_dict = None
        best_tac = -float('inf')
        best_epoch = 0

        for epoch in range(self.epochs):
            # 学习率调度
            if epoch < warmup_epochs:
                current_lr = initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                current_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))

            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            print(f"\nEpoch {epoch + 1}/{self.epochs} (lr: {current_lr:.2e})")
            epoch_start_time = datetime.now()

            # 训练一个 epoch
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            all_train_probs = []
            all_train_labels = []
            optimizer.zero_grad()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                outputs = model(images).squeeze(1)

                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, labels) / accumulation_steps

                loss.backward()

                # 累积指标
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    preds = (probs > LossConfig.TEST_THRESHOLD).float()
                    epoch_correct += (preds == labels).sum().item()
                    epoch_total += labels.size(0)
                    epoch_loss += loss.item() * accumulation_steps
                    all_train_probs.extend(probs.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())

                # 梯度累积
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # 处理剩余梯度
            if (batch_idx + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            batch_count = batch_idx + 1
            epoch_loss /= batch_count
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            try:
                epoch_auc = roc_auc_score(np.array(all_train_labels), np.array(all_train_probs))
            except ValueError:
                epoch_auc = 0.0

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            history['auc'].append(epoch_auc)

            epoch_time = datetime.now() - epoch_start_time
            print(f"训练完成 - 用时: {epoch_time.seconds}.{epoch_time.microseconds // 100000}s")

            # 评估测试集
            test_results = print_training_progress(
                epoch, self.epochs,
                {'loss': epoch_loss, 'accuracy': epoch_accuracy},
                test_loader, model, self.device
            )
            if test_results:
                test_loss, test_accuracy, test_auc, test_tac = test_results
                history['test_loss'].append(test_loss)
                history['test_accuracy'].append(test_accuracy)
                history['test_auc'].append(test_auc)
                history['test_tac'].append(test_tac)

                if test_tac > best_tac:
                    best_tac = test_tac
                    best_epoch = epoch + 1
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"✓ 发现更好的模型 (战术得分: {test_tac:.4f}, AUC: {test_auc:.4f})，已暂存到内存")

        # 保存模型
        print("\n保存模型到磁盘...")
        torch.save(model.state_dict(), os.path.join(self.model_save_dir, PathsConfig.FINAL_MODEL_FILENAME))
        print(f"✓ 最终模型已保存 ({PathsConfig.FINAL_MODEL_FILENAME})")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            torch.save(model.state_dict(), os.path.join(self.model_save_dir, PathsConfig.BEST_MODEL_FILENAME))
            print(f"✓ 最佳模型已保存 ({PathsConfig.BEST_MODEL_FILENAME}, 来自第{best_epoch}轮, 战术得分: {best_tac:.4f})")
        else:
            torch.save(model.state_dict(), os.path.join(self.model_save_dir, PathsConfig.BEST_MODEL_FILENAME))
            print(f"✓ 最佳模型已保存 ({PathsConfig.BEST_MODEL_FILENAME}, 与最终模型相同)")

        # 评估
        print("\n评估模型性能...")
        model.to(self.device)
        accuracy, class_report = self.evaluate_model(model, test_loader)

        # 生成报告
        print("\n生成完整训练报告...")
        self.save_complete_info(model, history, accuracy, class_report)
        print("✓ 完整训练报告已保存 (info.txt)")

        print("训练完成总结:")
        print(f"  - 最终训练准确率: {history['accuracy'][-1]:.4f}")
        print(f"  - 最终测试准确率: {history['test_accuracy'][-1]:.4f}")
        print(f"  - 实际训练轮次: {len(history['accuracy'])}")
        print(f"  - 模型保存路径: {self.model_save_dir}")

        return model, history


def test_class_weights():
    print("🧪 测试类别权重计算功能...")
    trainer = WiHelperTrainer()
    try:
        weights = trainer.compute_class_weights()
        print(f"✓ 类别权重计算成功: {weights}")
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


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

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-weights":
            test_class_weights()
            return
        elif sys.argv[1] in ["--help", "-h"]:
            print("用法:")
            print("  python train_model.py              # 完整训练")
            print("  python train_model.py --test-weights  # 测试类别权重计算")
            return

    trainer = WiHelperTrainer()
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
