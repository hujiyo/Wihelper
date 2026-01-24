#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#模型训练脚本 - 训练CNN模型来识别是否有目标

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TF的INFO和WARNING日志
import numpy as np
import tensorflow as tf
# 使用TensorFlow内置的keras而不是独立的keras包
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import tensorflow.keras.backend as K


def print_training_progress(epoch, total_epochs, logs, test_generator=None, model=None):
    """简单的训练进度显示函数"""
    print(f"\nEpoch {epoch + 1}/{total_epochs} - "
          f"损失: {logs.get('loss', 0):.4f} - 准确率: {logs.get('accuracy', 0):.4f}")

    # 使用测试集评估当前模型性能
    if test_generator is not None and model is not None:
        print("评估测试集...")
        test_generator.reset()  # 重置生成器
        y_pred_prob = model.predict(test_generator, verbose=0)
        y_true = test_generator.classes
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        test_loss, test_accuracy, test_auc = model.evaluate(test_generator, verbose=0)
        print(f"测试损失: {test_loss:.4f} - 测试准确率: {test_accuracy:.4f} - 测试AUC: {test_auc:.4f}")
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"混淆矩阵:  预测无目标  预测有目标")
        print(f"实际无目标    {tn:4d}        {fp:4d}")
        print(f"实际有目标    {fn:4d}        {tp:4d}")

        # 显示随机5个样本的预测值
        print("\n随机5个样本的预测详情:")

        # 随机选择5个样本
        np.random.seed(epoch)  # 使用epoch作为种子，保证每次运行结果一致
        sample_indices = np.random.choice(len(y_true), min(5, len(y_true)), replace=False)

        print("样本ID\t真实标签\t预测概率\t预测类别")
        print("-" * 45)
        for idx in sample_indices:
            true_label = y_true[idx]
            pred_prob = y_pred_prob[idx][0]  # 二分类的概率值
            pred_class = 1 if pred_prob > 0.5 else 0
            status = "✓" if true_label == pred_class else "✗"
            print(f"{idx:6d}\t{true_label}\t\t{pred_prob:.4f}\t\t{pred_class}\t\t{status}")

        print("-" * 50)

        return test_loss, test_accuracy, test_auc # 返回测试结果供后续使用
    return None

class WiHelperTrainer:
    def __init__(self, data_dir="image", model_save_dir="models"):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.img_height = 144 # 图像高度
        self.img_width = 144 # 图像宽度
        self.batch_size = 32 # 批次大小
        self.epochs = 50 # 最大轮次

        os.makedirs(model_save_dir, exist_ok=True) # 创建模型保存目录，如果目录不存在则创建

        # 配置GPU内存动态分配
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)# 设置GPU动态显存分配
            print(f"✓ 已启用GPU动态显存分配 ({len(physical_devices)}个GPU设备)")
        else:
            print("ℹ 未检测到GPU设备，使用CPU训练")
        
        np.random.seed(42) # 设置随机种子保证可重现性
        tf.random.set_seed(42)

    def create_model(self):
        """创建CNN模型"""
        print("\n" + "="*40)
        print("使用 CNN轻量极速版 (120×120内部处理)")
        print("="*40)
        return self._create_model()

    def _create_model(self):
        """CNN轻量极速版 (144×144输入，内部裁剪到120×120)"""
        # 144×144 → CenterCrop → 120×120 → 60×60 → 30×30 → 15×15
        # Flatten: 15×15×64 = 14,400
        model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),  # 144×144

            # 中心裁剪层: 144×144 → 120×120
            layers.CenterCrop(120, 120),

            # Block 1 (120 → 60)
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.1),

            # Block 2 (60 → 30)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.1),

            # Block 3 (30 → 15)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.15),

            # 15×15×64 = 14,400
            layers.Flatten(),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', keras.metrics.AUC(name='auc')])
        return model

    def add_noise_and_blur(self, image):
        # 随机选择增强类型，30%的概率添加噪声，30%的概率添加模糊，40%的概率不添加
        enhancement_type = np.random.choice(['noise', 'blur', 'none'], p=[0.3, 0.3, 0.4])

        if enhancement_type == 'noise': # 添加轻微高斯噪声
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image.astype(np.uint8), noise)
        elif enhancement_type == 'blur': # 添加轻微高斯模糊，模糊程度为3或5            
            ksize = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image.astype(np.uint8), (ksize, ksize), 0)

        return image.astype(np.float32)

    def compute_class_weights(self):
        """动态计算类别权重"""
        train_dir = os.path.join(self.data_dir, 'train')

        # 统计每个类别的样本数量
        nogot_count = len([f for f in os.listdir(os.path.join(train_dir, 'nogot'))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        got_count = len([f for f in os.listdir(os.path.join(train_dir, 'got'))
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        total_samples = nogot_count + got_count

        if total_samples == 0:
            print("警告: 训练数据为空，无法计算类别权重")
            return {0: 1.0, 1: 1.0}

        # 计算类别权重：权重 = 总样本数 / (类别数 * 该类别样本数)
        # 这样可以确保权重与类别频率成反比
        nogot_weight = total_samples / (2.0 * nogot_count)
        got_weight = total_samples / (2.0 * got_count)

        class_weights = {
            0: nogot_weight,  # nogot类别权重
            1: got_weight     # got类别权重
        }

        print("📊 类别权重计算:")
        print(f"  - nogot类别: {nogot_count}个样本, 权重: {nogot_weight:.4f}")
        print(f"  - got类别: {got_count}个样本, 权重: {got_weight:.4f}")
        print(f"  - 权重比值: {got_weight/nogot_weight:.4f}")

        return class_weights

    def create_data_generators(self):
        """创建数据生成器"""
        # 数据增强配置
        train_datagen = ImageDataGenerator(
            rescale=1./255, # 归一化
            brightness_range=[0.8, 1.2],      # 亮度调整 ±20%
            channel_shift_range=10.0,         # 颜色通道偏移
            horizontal_flip=True,             # 开启水平翻转
            # 暂时移除自定义噪声和模糊，先保证基础训练稳定
            # preprocessing_function=self.add_noise_and_blur 
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        # 训练数据生成器
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'), # 训练数据目录
            target_size=(self.img_height, self.img_width), # 目标大小
            batch_size=self.batch_size, # 批次大小
            class_mode='binary', # 二分类
            classes=['nogot', 'got'],  # 明确指定类别顺序: nogot=0(无目标), got=1(有目标)
            shuffle=True, # 打乱
            seed=42 # 随机种子
        )
        # 测试数据生成器
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'), # 测试数据目录
            target_size=(self.img_height, self.img_width), # 目标大小
            batch_size=self.batch_size, # 批次大小
            class_mode='binary', # 二分类
            classes=['nogot', 'got'],  # 明确指定类别顺序: nogot=0(无目标), got=1(有目标)
            shuffle=False # 不打乱
        )
        return train_generator, test_generator

    def save_complete_info(self, model, history, accuracy, class_report):
        info_path = os.path.join(self.model_save_dir, 'info.txt') # 信息保存路径

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("基本信息:\n")
            f.write(f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最终测试准确率: {accuracy:.4f}\n")
            f.write(f"实际训练轮次: {len(history['accuracy'])}\n\n")
            f.write(f"图像大小: {self.img_height}x{self.img_width}\n")
            f.write(f"批次大小: {self.batch_size}\n")
            f.write(f"最大轮次: {self.epochs}\n")
            f.write(f"优化器: adam (initial_lr=1e-4)\n")
            f.write(f"损失函数: binary_crossentropy + class_weight\n")
            f.write(f"指标: accuracy, auc\n\n")
            f.write("类别权重信息 (仅供参考，未在Loss中叠加):\n")
            f.write(f"nogot类别权重: {self.class_weights[0]:.4f}\n")
            f.write(f"got类别权重: {self.class_weights[1]:.4f}\n")
            f.write(f"权重比值 (got/nogot): {self.class_weights[1]/self.class_weights[0]:.4f}\n\n")
            f.write("-" * 30 + "\n") # 模型结构分隔线
            stringlist = [] # 模型结构列表
            model.summary(print_fn=lambda x: stringlist.append(x))
            for line in stringlist:
                f.write(line + "\n")
            f.write("\n")

            f.write("训练历史:\n")
            f.write(f"总训练轮次: {len(history['accuracy'])}\n\n")            
            f.write("轮次\t训练准确率\t测试准确率\t训练损失\t测试损失\t训练AUC\t测试AUC\n")
            f.write("-" * 80 + "\n") # 表头分隔线
            
            for epoch in range(len(history['accuracy'])): # 每轮数据
                f.write(f"{epoch+1:2d}\t")
                f.write(f"{history['accuracy'][epoch]:.4f}\t")
                f.write(f"{history['test_accuracy'][epoch]:.4f}\t")
                f.write(f"{history['loss'][epoch]:.4f}\t")
                f.write(f"{history['test_loss'][epoch]:.4f}\t")
                f.write(f"{history['auc'][epoch]:.4f}\t")
                f.write(f"{history['test_auc'][epoch]:.4f}\t")
                f.write("\n")

            f.write("\n" + "=" * 50 + "\n最终结果总结:\n" + "=" * 50 + "\n")
            f.write(f"最终训练准确率: {history['accuracy'][-1]:.4f}\n")
            f.write(f"最终测试准确率: {history['test_accuracy'][-1]:.4f}\n")
            f.write(f"最终训练损失: {history['loss'][-1]:.4f}\n")
            f.write(f"最终测试损失: {history['test_loss'][-1]:.4f}\n")
            f.write(f"最终训练AUC: {history['auc'][-1]:.4f}\n")
            f.write(f"最终测试AUC: {history['test_auc'][-1]:.4f}\n\n")

            f.write("分类报告:\n")
            f.write("-" * 30 + "\n")
            f.write(class_report)
            f.write("\n")

    def evaluate_model(self, model, test_generator):
        print("📊 模型评估")
        # 预测
        y_pred_prob = model.predict(test_generator) # 预测
        y_pred = (y_pred_prob > 0.5).astype(int).flatten() # 预测结果
        y_true = test_generator.classes # 真实结果

        # 分类报告
        class_names = ['无目标', '有目标']
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("分类报告:" + report)
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print("\n混淆矩阵:\n" + str(cm))
        # 计算准确率
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print(f"准确率: {accuracy:.4f}")

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()

        tick_marks = [0, 1]
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # 添加数值标签
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

        # 检查数据目录
        if not os.path.exists(self.data_dir):
            print("错误: 数据目录不存在")
            print(f"路径: {self.data_dir}")
            return

        # 创建模型
        print("\n步骤1/3: 创建模型...")
        model = self.create_model()
        model.summary() # 打印模型结构

        # 开始训练
        return self._train(model)

    def _train(self, model):
        """CNN模型训练流程 - 使用动态类别权重"""
        print("\n步骤2/3: 加载训练数据...")
        train_generator, test_generator = self.create_data_generators()
        print("数据统计:")
        print(f"训练样本数: {train_generator.samples}")
        print(f"验证样本数: {test_generator.samples}")
        print(f"类别映射: {train_generator.class_indices}")

        # 计算类别权重
        self.class_weights = self.compute_class_weights()

        print("\n步骤3/3: 开始训练模型...")
        print(f"  - 最大轮次: {self.epochs}")
        print(f"  - 批次大小: {self.batch_size}")
        print("  - 优化器: Adam")
        print("  - 损失函数: Binary Cross Entropy + 动态类别权重")
        print("  - 指标: Accuracy, AUC")
        print("\n开始训练 (每轮显示进度和测试评估)...")

        # 学习率调度参数
        warmup_epochs = 5
        initial_lr = 1e-4
        peak_lr = 3e-4
        min_lr = 1e-6
        accumulation_steps = 4
        patience = 15
        patience_counter = 0

        # 训练历史记录
        history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'test_loss': [], 'test_accuracy': [], 'test_auc': []
        }
        best_weights = None
        best_auc = 0.0
        best_epoch = 0

        for epoch in range(self.epochs):
            # 计算当前学习率
            if epoch < warmup_epochs:
                current_lr = initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                current_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))

            # 应用学习率
            tf.keras.backend.set_value(model.optimizer.learning_rate, current_lr)

            print(f"\nEpoch {epoch + 1}/{self.epochs} (lr: {current_lr:.2e})")
            epoch_start_time = datetime.now()

            # 训练一个epoch - 使用梯度累积
            epoch_logs = {'loss': 0, 'accuracy': 0, 'auc': 0}
            batch_count = 0
            accum_count = 0
            accumulated_gradients = None
            accum_labels = []
            accum_batches = []
            train_generator.reset()

            for batch_x, batch_y in train_generator:
                accum_batches.append((batch_x, batch_y))
                accum_labels.extend(batch_y.tolist())
                accum_count += 1

                if accum_count >= accumulation_steps:
                    # 计算动态权重
                    accum_labels_arr = np.array(accum_labels)
                    n_pos = np.sum(accum_labels_arr == 1)
                    n_neg = np.sum(accum_labels_arr == 0)
                    total = n_pos + n_neg

                    if n_pos > 0 and n_neg > 0:
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                    else:
                        pos_weight = 1.0
                        neg_weight = 1.0

                    for acc_batch_x, acc_batch_y in accum_batches:
                        batch_weights = np.array([pos_weight if label == 1 else neg_weight for label in acc_batch_y])

                        with tf.GradientTape() as tape:
                            predictions = model(acc_batch_x, training=True)
                            bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                            loss = bce(acc_batch_y, predictions[:, 0])
                            weighted_loss = loss * batch_weights
                            batch_loss = tf.reduce_mean(weighted_loss) / accumulation_steps

                        gradients = tape.gradient(batch_loss, model.trainable_variables)

                        if accumulated_gradients is None:
                            accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) if g is not None else None for g in gradients]

                        for i, grad in enumerate(gradients):
                            if grad is not None and accumulated_gradients[i] is not None:
                                accumulated_gradients[i].assign_add(grad)

                        epoch_logs['loss'] += float(tf.reduce_mean(loss))
                        pred_classes = tf.cast(predictions[:, 0] > 0.5, tf.float32)
                        epoch_logs['accuracy'] += float(tf.reduce_mean(tf.cast(tf.equal(pred_classes, acc_batch_y), tf.float32)))
                        epoch_logs['auc'] += float(tf.keras.metrics.AUC()(acc_batch_y, predictions[:, 0]))
                        batch_count += 1

                    # 应用梯度
                    model.optimizer.apply_gradients(
                        [(g, v) for g, v in zip(accumulated_gradients, model.trainable_variables) if g is not None]
                    )

                    # 重置
                    for g in accumulated_gradients:
                        if g is not None:
                            g.assign(tf.zeros_like(g))
                    accum_count = 0
                    accum_labels = []
                    accum_batches = []

                if batch_count >= len(train_generator):
                    break

            # 处理剩余数据
            if accum_count > 0 and len(accum_batches) > 0:
                accum_labels_arr = np.array(accum_labels)
                n_pos = np.sum(accum_labels_arr == 1)
                n_neg = np.sum(accum_labels_arr == 0)
                total = n_pos + n_neg

                if n_pos > 0 and n_neg > 0:
                    pos_weight = total / (2.0 * n_pos)
                    neg_weight = total / (2.0 * n_neg)
                else:
                    pos_weight = 1.0
                    neg_weight = 1.0

                for acc_batch_x, acc_batch_y in accum_batches:
                    batch_weights = np.array([pos_weight if label == 1 else neg_weight for label in acc_batch_y])

                    with tf.GradientTape() as tape:
                        predictions = model(acc_batch_x, training=True)
                        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                        loss = bce(acc_batch_y, predictions[:, 0])
                        weighted_loss = loss * batch_weights
                        batch_loss = tf.reduce_mean(weighted_loss) / len(accum_batches)

                    gradients = tape.gradient(batch_loss, model.trainable_variables)

                    if accumulated_gradients is None:
                        accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) if g is not None else None for g in gradients]

                    for i, grad in enumerate(gradients):
                        if grad is not None and accumulated_gradients[i] is not None:
                            accumulated_gradients[i].assign_add(grad)

                    epoch_logs['loss'] += float(tf.reduce_mean(loss))
                    pred_classes = tf.cast(predictions[:, 0] > 0.5, tf.float32)
                    epoch_logs['accuracy'] += float(tf.reduce_mean(tf.cast(tf.equal(pred_classes, acc_batch_y), tf.float32)))
                    epoch_logs['auc'] += float(tf.keras.metrics.AUC()(acc_batch_y, predictions[:, 0]))
                    batch_count += 1

                if accumulated_gradients is not None:
                    model.optimizer.apply_gradients(
                        [(g, v) for g, v in zip(accumulated_gradients, model.trainable_variables) if g is not None]
                    )

            # 计算平均值
            for key in epoch_logs:
                epoch_logs[key] /= batch_count

            history['loss'].append(epoch_logs['loss'])
            history['accuracy'].append(epoch_logs['accuracy'])
            history['auc'].append(epoch_logs['auc'])

            epoch_time = datetime.now() - epoch_start_time
            print(f"训练完成 - 用时: {epoch_time.seconds}.{epoch_time.microseconds//100000}s")

            # 评估测试集
            test_results = print_training_progress(epoch, self.epochs, epoch_logs, test_generator, model)
            if test_results:
                test_loss, test_accuracy, test_auc = test_results
                history['test_loss'].append(test_loss)
                history['test_accuracy'].append(test_accuracy)
                history['test_auc'].append(test_auc)

                # 早停
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_weights = model.get_weights()
                    print(f"✓ 发现更好的模型 (AUC: {test_auc:.4f})，已暂存到内存")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停: 测试AUC在{patience}轮内没有提升")
                        break

        # 保存模型
        print("\n保存模型到磁盘...")
        final_weights = model.get_weights()
        model.save(os.path.join(self.model_save_dir, 'final_model.h5'))
        print("✓ 最终模型已保存 (final_model.h5)")

        if best_weights is not None:
            model.set_weights(best_weights)
            model.save(os.path.join(self.model_save_dir, 'best_model.h5'))
            print(f"✓ 最佳模型已保存 (best_model.h5, 来自第{best_epoch}轮, AUC: {best_auc:.4f})")
            model.set_weights(final_weights)
        else:
            model.save(os.path.join(self.model_save_dir, 'best_model.h5'))
            print("✓ 最佳模型已保存 (best_model.h5, 与最终模型相同)")

        # 评估
        print("\n评估模型性能...")
        accuracy, class_report = self.evaluate_model(model, test_generator)

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

    def save_complete_info(self, model, history, accuracy, class_report):
        """保存CNN训练信息"""
        self.save_complete_info(model, history, accuracy, class_report)

    def save_complete_info(self, model, history, accuracy, class_report):
        """保存完整训练信息"""
        info_path = os.path.join(self.model_save_dir, 'info.txt')

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("基本信息:\n")
            f.write(f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最终测试准确率: {accuracy:.4f}\n")
            f.write(f"实际训练轮次: {len(history['accuracy'])}\n\n")
            f.write(f"图像大小: {self.img_height}x{self.img_width}\n")
            f.write(f"批次大小: {self.batch_size}\n")
            f.write(f"最大轮次: {self.epochs}\n")
            f.write(f"优化器: adam (initial_lr=1e-4)\n")
            f.write(f"损失函数: binary_crossentropy + class_weight\n")
            f.write(f"指标: accuracy, auc\n\n")
            f.write("类别权重信息 (仅供参考，未在Loss中叠加):\n")
            f.write(f"nogot类别权重: {self.class_weights[0]:.4f}\n")
            f.write(f"got类别权重: {self.class_weights[1]:.4f}\n")
            f.write(f"权重比值 (got/nogot): {self.class_weights[1]/self.class_weights[0]:.4f}\n\n")
            f.write("-" * 30 + "\n") # 模型结构分隔线
            stringlist = [] # 模型结构列表
            model.summary(print_fn=lambda x: stringlist.append(x))
            for line in stringlist:
                f.write(line + "\n")
            f.write("\n")

            f.write("训练历史:\n")
            f.write(f"总训练轮次: {len(history['accuracy'])}\n\n")
            f.write("轮次\t训练准确率\t测试准确率\t训练损失\t测试损失\t训练AUC\t测试AUC\n")
            f.write("-" * 80 + "\n") # 表头分隔线

            for epoch in range(len(history['accuracy'])): # 每轮数据
                f.write(f"{epoch+1:2d}\t")
                f.write(f"{history['accuracy'][epoch]:.4f}\t")
                f.write(f"{history['test_accuracy'][epoch]:.4f}\t")
                f.write(f"{history['loss'][epoch]:.4f}\t")
                f.write(f"{history['test_loss'][epoch]:.4f}\t")
                f.write(f"{history['auc'][epoch]:.4f}\t")
                f.write(f"{history['test_auc'][epoch]:.4f}\t")
                f.write("\n")

            f.write("\n" + "=" * 50 + "\n最终结果总结:\n" + "=" * 50 + "\n")
            f.write(f"最终训练准确率: {history['accuracy'][-1]:.4f}\n")
            f.write(f"最终测试准确率: {history['test_accuracy'][-1]:.4f}\n")
            f.write(f"最终训练损失: {history['loss'][-1]:.4f}\n")
            f.write(f"最终测试损失: {history['test_loss'][-1]:.4f}\n")
            f.write(f"最终训练AUC: {history['auc'][-1]:.4f}\n")
            f.write(f"最终测试AUC: {history['test_auc'][-1]:.4f}\n\n")

            f.write("分类报告:\n")
            f.write("-" * 30 + "\n")
            f.write(class_report)
            f.write("\n")

        # 学习率调度参数
        warmup_epochs = 5           # 预热轮次
        initial_lr = 1e-4           # 预热起始学习率
        peak_lr = 3e-4              # 峰值学习率
        min_lr = 1e-6               # 最小学习率
        accumulation_steps = 4      # 梯度累积步数
        patience = 15
        patience_counter = 0
        
        # 训练历史记录
        history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'test_loss': [], 'test_accuracy': [], 'test_auc': []
        }
        best_weights = None
        best_auc = 0.0
        best_epoch = 0

        for epoch in range(self.epochs):
            # 计算当前学习率
            if epoch < warmup_epochs:
                # 预热阶段：线性增加学习率
                current_lr = initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
            else:
                # 预热后：余弦退火衰减
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                current_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))
            
            # 应用学习率
            tf.keras.backend.set_value(model.optimizer.learning_rate, current_lr)
            
            print(f"\nEpoch {epoch + 1}/{self.epochs} (lr: {current_lr:.2e})")
            epoch_start_time = datetime.now()

            # 训练一个epoch - 使用梯度累积
            epoch_logs = {'loss': 0, 'accuracy': 0, 'auc': 0}
            batch_count = 0
            accum_count = 0  # 累积计数器
            accumulated_gradients = None  # 累积的梯度
            accum_labels = []  # 累积周期内的标签，用于动态权重计算
            accum_batches = []  # 累积周期内的batch数据
            train_generator.reset() #重置生成器重新开始

            for batch_x, batch_y in train_generator:
                # 收集当前累积周期的数据
                accum_batches.append((batch_x, batch_y))
                accum_labels.extend(batch_y.tolist())
                accum_count += 1
                
                # 达到累积步数时，计算动态权重并处理梯度
                if accum_count >= accumulation_steps:
                    # 在累积周期内计算动态权重
                    accum_labels_arr = np.array(accum_labels)
                    n_pos = np.sum(accum_labels_arr == 1)
                    n_neg = np.sum(accum_labels_arr == 0)
                    total = n_pos + n_neg
                    
                    # 计算动态权重（防止除零）
                    if n_pos > 0 and n_neg > 0:
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                    else:
                        # 极端情况：全正或全负，使用默认权重
                        pos_weight = 1.0
                        neg_weight = 1.0
                    
                    # 对累积周期内的每个batch计算梯度
                    for acc_batch_x, acc_batch_y in accum_batches:
                        # 为当前batch构建样本权重
                        batch_weights = np.array([pos_weight if label == 1 else neg_weight for label in acc_batch_y])
                        
                        with tf.GradientTape() as tape:
                            predictions = model(acc_batch_x, training=True)
                            # 使用标准BCE
                            bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                            loss = bce(acc_batch_y, predictions[:, 0])
                            # 应用动态权重
                            weighted_loss = loss * batch_weights
                            batch_loss = tf.reduce_mean(weighted_loss) / accumulation_steps
                        
                        # 计算梯度
                        gradients = tape.gradient(batch_loss, model.trainable_variables)
                        
                        # 累积梯度
                        if accumulated_gradients is None:
                            accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) if g is not None else None for g in gradients]
                        
                        for i, grad in enumerate(gradients):
                            if grad is not None and accumulated_gradients[i] is not None:
                                accumulated_gradients[i].assign_add(grad)
                        
                        # 记录指标
                        epoch_logs['loss'] += float(tf.reduce_mean(loss))
                        pred_classes = tf.cast(predictions[:, 0] > 0.5, tf.float32)
                        epoch_logs['accuracy'] += float(tf.reduce_mean(tf.cast(tf.equal(pred_classes, acc_batch_y), tf.float32)))
                        epoch_logs['auc'] += float(tf.keras.metrics.AUC()(acc_batch_y, predictions[:, 0]))
                        batch_count += 1
                    
                    # 应用累积的梯度
                    model.optimizer.apply_gradients(
                        [(g, v) for g, v in zip(accumulated_gradients, model.trainable_variables) if g is not None]
                    )
                    
                    # 重置累积状态
                    for g in accumulated_gradients:
                        if g is not None:
                            g.assign(tf.zeros_like(g))
                    accum_count = 0
                    accum_labels = []
                    accum_batches = []

                # 避免无限循环
                if batch_count >= len(train_generator):
                    break
            
            # 处理剩余的累积数据 (如果有)
            if accum_count > 0 and len(accum_batches) > 0:
                accum_labels_arr = np.array(accum_labels)
                n_pos = np.sum(accum_labels_arr == 1)
                n_neg = np.sum(accum_labels_arr == 0)
                total = n_pos + n_neg
                
                if n_pos > 0 and n_neg > 0:
                    pos_weight = total / (2.0 * n_pos)
                    neg_weight = total / (2.0 * n_neg)
                else:
                    pos_weight = 1.0
                    neg_weight = 1.0
                
                for acc_batch_x, acc_batch_y in accum_batches:
                    batch_weights = np.array([pos_weight if label == 1 else neg_weight for label in acc_batch_y])
                    
                    with tf.GradientTape() as tape:
                        predictions = model(acc_batch_x, training=True)
                        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                        loss = bce(acc_batch_y, predictions[:, 0])
                        weighted_loss = loss * batch_weights
                        batch_loss = tf.reduce_mean(weighted_loss) / len(accum_batches)
                    
                    gradients = tape.gradient(batch_loss, model.trainable_variables)
                    
                    if accumulated_gradients is None:
                        accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) if g is not None else None for g in gradients]
                    
                    for i, grad in enumerate(gradients):
                        if grad is not None and accumulated_gradients[i] is not None:
                            accumulated_gradients[i].assign_add(grad)
                    
                    epoch_logs['loss'] += float(tf.reduce_mean(loss))
                    pred_classes = tf.cast(predictions[:, 0] > 0.5, tf.float32)
                    epoch_logs['accuracy'] += float(tf.reduce_mean(tf.cast(tf.equal(pred_classes, acc_batch_y), tf.float32)))
                    epoch_logs['auc'] += float(tf.keras.metrics.AUC()(acc_batch_y, predictions[:, 0]))
                    batch_count += 1
                
                if accumulated_gradients is not None:
                    model.optimizer.apply_gradients(
                        [(g, v) for g, v in zip(accumulated_gradients, model.trainable_variables) if g is not None]
                    )

            # 计算平均值
            for key in epoch_logs:
                epoch_logs[key] /= batch_count

            # 保存训练历史
            history['loss'].append(epoch_logs['loss'])
            history['accuracy'].append(epoch_logs['accuracy'])
            history['auc'].append(epoch_logs['auc'])

            epoch_time = datetime.now() - epoch_start_time
            print(f"训练完成 - 用时: {epoch_time.seconds}.{epoch_time.microseconds//100000}s")

            # 显示进度和评估测试集
            test_results = print_training_progress(epoch, self.epochs, epoch_logs, test_generator, model)
            if test_results:
                test_loss, test_accuracy, test_auc = test_results
                history['test_loss'].append(test_loss)
                history['test_accuracy'].append(test_accuracy)
                history['test_auc'].append(test_auc)

                # 早停逻辑 - 基于AUC而非准确率
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # 将最佳模型权重保存到内存
                    best_weights = model.get_weights()
                    print(f"✓ 发现更好的模型 (AUC: {test_auc:.4f})，已暂存到内存")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停: 测试AUC在{patience}轮内没有提升")
                        break

        # 训练结束后统一保存模型
        print("\n保存模型到磁盘...")
        
        # 保存最终模型 (当前训练结束时的模型)
        final_weights = model.get_weights()  # 先保存最终权重
        model.save(os.path.join(self.model_save_dir, 'final_model.h5'))
        print("✓ 最终模型已保存 (final_model.h5)")
        
        # 保存最佳模型 (从内存中恢复最佳权重后保存)
        if best_weights is not None:
            model.set_weights(best_weights)  # 恢复最佳权重
            model.save(os.path.join(self.model_save_dir, 'best_model.h5'))
            print(f"✓ 最佳模型已保存 (best_model.h5, 来自第{best_epoch}轮, AUC: {best_auc:.4f})")
            model.set_weights(final_weights)  # 恢复最终权重用于后续评估
        else:
            # 如果没有最佳模型（第一轮就停止了），使用最终模型作为最佳模型
            model.save(os.path.join(self.model_save_dir, 'best_model.h5'))
            print("✓ 最佳模型已保存 (best_model.h5, 与最终模型相同)")

        # 评估模型
        print("\n评估模型性能...")
        accuracy, class_report = self.evaluate_model(model, test_generator)

        # 生成完整训练报告
        print("\n生成完整训练报告...")
        self.save_complete_info(model, history, accuracy, class_report)
        print("✓ 完整训练报告已保存 (info.txt)")

        print("训练完成总结:")
        print(f"  - 最终训练准确率: {history['accuracy'][-1]:.4f}")
        print(f"  - 最终测试准确率: {history['test_accuracy'][-1]:.4f}")
        print(f"  - 实际训练轮次: {len(history['accuracy'])}")
        print(f"  - 模型保存路径: {self.model_save_dir}")
        print("  - 文件列表:")
        print("    * final_model.h5 (完整模型)")
        print("    * best_model.h5 (最佳模型)")
        print("    * info.txt (完整训练报告)")
        print("    * confusion_matrix.png (混淆矩阵)")
        
        return model, history

def test_class_weights():
    """测试类别权重计算功能"""
    print("🧪 测试类别权重计算功能...")
    trainer = WiHelperTrainer()

    try:
        # 测试类别权重计算
        weights = trainer.compute_class_weights()
        print(f"✓ 类别权重计算成功: {weights}")

        # 验证权重计算逻辑
        train_dir = os.path.join(trainer.data_dir, 'train')
        nogot_count = len([f for f in os.listdir(os.path.join(train_dir, 'nogot'))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        got_count = len([f for f in os.listdir(os.path.join(train_dir, 'got'))
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        total_samples = nogot_count + got_count
        expected_nogot_weight = total_samples / (2.0 * nogot_count)
        expected_got_weight = total_samples / (2.0 * got_count)

        print("验证计算结果:")
        print(f"  - 期望nogot权重: {expected_nogot_weight:.4f}, 实际: {weights[0]:.4f}")
        print(f"  - 期望got权重: {expected_got_weight:.4f}, 实际: {weights[1]:.4f}")

        if abs(weights[0] - expected_nogot_weight) < 1e-6 and abs(weights[1] - expected_got_weight) < 1e-6:
            print("✓ 权重计算验证通过")
        else:
            print("✗ 权重计算验证失败")

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-weights":
            test_class_weights()
            return
        elif sys.argv[1] in ["--help", "-h"]:
            print("用法:")
            print("  python train_model.py              # 完整训练")
            print("  python train_model.py --test-weights  # 测试类别权重计算")
            return
    
    # 默认: 完整训练流程
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
