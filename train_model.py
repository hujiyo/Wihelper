#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#模型训练脚本 - 训练CNN模型来识别是否有目标

import os
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


class WiHelperCNN(nn.Module):
    """空洞卷积版 - 全程保持空间分辨率，中心裁剪后压缩"""
    def __init__(self):
        super().__init__()
        # 空洞卷积特征提取: 120×120 全程不变
        # Block 1: dilation=1, 感受野 3→5
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Block 2: dilation=2, 感受野 5→9→13
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Block 3: dilation=4, 感受野 13→21→29
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 中心92×92 → 压缩到 12×12
        self.compress = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 分类头 12×12×32 = 4608
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.block1(x)   # [B, 32, 120, 120]
        x = self.block2(x)   # [B, 64, 120, 120]
        x = self.block3(x)   # [B, 64, 120, 120]
        # 中心裁剪64×64
        h = (x.shape[2] - 92) // 2
        w = (x.shape[3] - 92) // 2
        x = x[:, :, h:h+92, w:w+92]  # [B, 64, 92, 92]
        x = self.compress(x)    # [B, 32, 8, 8]
        x = self.classifier(x)  # [B, 1]
        return x


def tactical_score(y_true, y_pred_prob, threshold=0.5, fp_penalty=5.0):
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
                preds = (probs > 0.5).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        y_true = np.array(all_labels)
        y_pred_prob = np.array(all_probs)
        y_pred = (y_pred_prob > 0.5).astype(int)

        test_loss = test_loss_total / test_total
        test_accuracy = test_correct / test_total
        try:
            test_auc = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            test_auc = 0.0
        test_tac = tactical_score(y_true, y_pred_prob)

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
            pred_class = 1 if pred_prob > 0.5 else 0
            status = "✓" if true_label == pred_class else "✗"
            print(f"{idx:6d}\t{true_label}\t\t{pred_prob:.4f}\t\t{pred_class}\t\t{status}")
        print("-" * 50)

        return test_loss, test_accuracy, test_auc, test_tac
    return None


class CenterCrop:
    """中心裁剪 144→120"""
    def __init__(self, crop_size=120):
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
    LABEL_MAP = {'got': 1, 'nogot': 0}

    def __init__(self, root, transform_pil, target_size=(120, 120)):
        super().__init__(root)
        self.transform_pil = transform_pil
        self.target_size = target_size
        self.center_crop = CenterCrop(120)

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
    def __init__(self, data_dir="image", model_save_dir="models"):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.raw_height = 144
        self.raw_width = 144
        self.img_height = 120
        self.img_width = 120
        self.batch_size = 32
        self.batches_per_epoch = 100
        self.epochs = 50

        os.makedirs(model_save_dir, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    def create_model(self):
        print("\n" + "=" * 40)
        print("使用 CNN轻量极速版 (120×120输入)")
        print("=" * 40)
        model = WiHelperCNN().to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        return model

    def add_noise_and_blur(self, image):
        enhancement_type = np.random.choice(['noise', 'blur', 'none'], p=[0.3, 0.3, 0.4])
        if enhancement_type == 'noise':
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image.astype(np.uint8), noise)
        elif enhancement_type == 'blur':
            ksize = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image.astype(np.uint8), (ksize, ksize), 0)
        return image.astype(np.float32)

    def compute_class_weights(self):
        train_dir = os.path.join(self.data_dir, 'train')
        nogot_count = len([f for f in os.listdir(os.path.join(train_dir, 'nogot'))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        got_count = len([f for f in os.listdir(os.path.join(train_dir, 'got'))
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
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
            os.path.join(self.data_dir, 'train'),
            transform_pil=train_transform,
            target_size=(self.img_height, self.img_width),
        )
        test_dataset = TrainDataset(
            os.path.join(self.data_dir, 'test'),
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
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # 注意: ImageFolder 原始映射是 {'got':0, 'nogot':1}
        # TrainDataset.__getitem__ 已翻转为 nogot=0, got=1
        print(f"类别映射: nogot=0(无目标), got=1(有目标)")
        return train_loader, test_loader

    def save_complete_info(self, model, history, accuracy, class_report):
        info_path = os.path.join(self.model_save_dir, 'info.txt')
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
        y_pred = (y_pred_prob > 0.5).astype(int)

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
        plt.savefig(os.path.join(self.model_save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
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

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        warmup_epochs = 5
        initial_lr = 1e-4
        peak_lr = 3e-4
        min_lr = 1e-6
        accumulation_steps = 4
        patience = 15
        patience_counter = 0

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
                    preds = (probs > 0.5).float()
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
                    patience_counter = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"✓ 发现更好的模型 (战术得分: {test_tac:.4f}, AUC: {test_auc:.4f})，已暂存到内存")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停: 战术得分在{patience}轮内没有提升")
                        break

        # 保存模型
        print("\n保存模型到磁盘...")
        torch.save(model.state_dict(), os.path.join(self.model_save_dir, 'final_model.pth'))
        print("✓ 最终模型已保存 (final_model.pth)")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            torch.save(model.state_dict(), os.path.join(self.model_save_dir, 'best_model.pth'))
            print(f"✓ 最佳模型已保存 (best_model.pth, 来自第{best_epoch}轮, 战术得分: {best_tac:.4f})")
        else:
            torch.save(model.state_dict(), os.path.join(self.model_save_dir, 'best_model.pth'))
            print("✓ 最佳模型已保存 (best_model.pth, 与最终模型相同)")

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
