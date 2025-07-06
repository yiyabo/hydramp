#!/usr/bin/env python3
"""
使用Gram-.fasta数据进行HydrAMP模型微调训练的实用脚本
这个方案基于预训练模型进行微调，避免从零训练的困难
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 设置GPU配置
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8),
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# 导入HydrAMP模块
import amp.data_utils.sequence as du_sequence
from amp.config import MIN_LENGTH, MAX_LENGTH, LATENT_DIM, MIN_KL, RCL_WEIGHT, HIDDEN_DIM, MAX_TEMPERATURE
from amp.utils import basic_model_serializer
from keras import backend
from keras.optimizers import Adam

def prepare_gram_data(fasta_file):
    """
    准备Gram-.fasta数据用于训练
    """
    print(f"正在处理FASTA文件: {fasta_file}")
    
    # 定义标准氨基酸
    STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
    
    def check_if_std_aa(seq):
        return all(aa in STANDARD_AA for aa in seq.upper())
    
    # 解析FASTA文件
    sequences = []
    ids = []
    current_id = ''
    current_seq = ''
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    ids.append(current_id)
                current_id = line[1:]
                current_seq = ''
            else:
                current_seq += line
        
        # 添加最后一个序列
        if current_seq:
            sequences.append(current_seq)
            ids.append(current_id)
    
    # 过滤序列：长度在范围内且只包含标准氨基酸
    filtered_sequences = []
    for seq in sequences:
        if MIN_LENGTH <= len(seq) <= MAX_LENGTH and check_if_std_aa(seq):
            filtered_sequences.append(seq)
    
    print(f"原始序列数: {len(sequences)}")
    print(f"有效序列数: {len(filtered_sequences)}")
    
    # 转换为one-hot编码并padding
    one_hot_sequences = du_sequence.pad(du_sequence.to_one_hot(filtered_sequences))
    labels = np.ones(len(filtered_sequences))  # 全部标记为正样本（AMP）
    
    return one_hot_sequences, labels, filtered_sequences

def fine_tune_with_gram_data(gram_sequences, gram_labels, original_sequences):
    """
    使用Gram数据对预训练模型进行微调
    """
    print("开始微调训练...")
    
    # 1. 加载预训练的分类器
    print("加载预训练分类器...")
    bms = basic_model_serializer.BasicModelSerializer()
    
    try:
        amp_classifier = bms.load_model('models/amp_classifier')
        mic_classifier = bms.load_model('models/mic_classifier')
        print("✅ 成功加载预训练分类器")
    except Exception as e:
        print(f"❌ 无法加载预训练分类器: {e}")
        return False
    
    # 2. 使用预训练分类器预测Gram数据的标签
    print("预测Gram数据的AMP和MIC概率...")
    amp_classifier_model = amp_classifier()
    mic_classifier_model = mic_classifier()
    
    gram_amp_probs = amp_classifier_model.predict(gram_sequences, verbose=1).flatten()
    gram_mic_probs = mic_classifier_model.predict(gram_sequences, verbose=1).flatten()
    
    print(f"Gram数据AMP概率 - 平均: {gram_amp_probs.mean():.4f}, 范围: [{gram_amp_probs.min():.4f}, {gram_amp_probs.max():.4f}]")
    print(f"Gram数据MIC概率 - 平均: {gram_mic_probs.mean():.4f}, 范围: [{gram_mic_probs.min():.4f}, {gram_mic_probs.max():.4f}]")
    
    # 3. 过滤高质量序列用于微调
    # 选择AMP概率高的序列进行微调
    high_quality_mask = gram_amp_probs > 0.7  # 只使用高质量预测的序列
    
    if high_quality_mask.sum() < 10:
        print("⚠️  高质量序列太少，降低阈值...")
        high_quality_mask = gram_amp_probs > 0.5
    
    if high_quality_mask.sum() < 5:
        print("⚠️  高质量序列仍然太少，使用所有序列...")
        high_quality_mask = np.ones(len(gram_sequences), dtype=bool)
    
    filtered_sequences = gram_sequences[high_quality_mask]
    filtered_amp_probs = gram_amp_probs[high_quality_mask]
    filtered_mic_probs = gram_mic_probs[high_quality_mask]
    filtered_original_seqs = [original_sequences[i] for i in range(len(original_sequences)) if high_quality_mask[i]]
    
    print(f"用于微调的高质量序列数: {len(filtered_sequences)}")
    
    # 4. 数据增强：为少量数据创建变体
    print("进行数据增强...")
    augmented_sequences = []
    augmented_amp_probs = []
    augmented_mic_probs = []
    
    for seq, amp_prob, mic_prob in zip(filtered_sequences, filtered_amp_probs, filtered_mic_probs):
        # 原始序列
        augmented_sequences.append(seq)
        augmented_amp_probs.append(amp_prob)
        augmented_mic_probs.append(mic_prob)
        
        # 添加轻微噪声的变体（只对高质量序列）
        if amp_prob > 0.8:
            for _ in range(3):  # 每个高质量序列创建3个变体
                noisy_seq = add_sequence_noise(seq)
                if noisy_seq is not None:
                    augmented_sequences.append(noisy_seq)
                    augmented_amp_probs.append(amp_prob * 0.95)  # 略微降低概率
                    augmented_mic_probs.append(mic_prob * 0.95)
    
    augmented_sequences = np.array(augmented_sequences)
    augmented_amp_probs = np.array(augmented_amp_probs)
    augmented_mic_probs = np.array(augmented_mic_probs)
    
    print(f"数据增强后序列数: {len(augmented_sequences)}")
    
    # 5. 简化的微调：只训练分类器头
    print("开始微调分类器...")
    
    # 分割训练和验证数据
    train_seqs, val_seqs, train_amp, val_amp, train_mic, val_mic = train_test_split(
        augmented_sequences, augmented_amp_probs, augmented_mic_probs, 
        test_size=0.2, random_state=42
    )
    
    # 微调AMP分类器
    print("微调AMP分类器...")
    amp_model = amp_classifier_model
    
    # 冻结除最后几层外的所有层
    for layer in amp_model.layers[:-2]:
        layer.trainable = False
    
    # 重新编译模型
    amp_model.compile(
        optimizer=Adam(lr=1e-4),  # 使用较小的学习率
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练
    history_amp = amp_model.fit(
        train_seqs, train_amp,
        validation_data=(val_seqs, val_amp),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # 微调MIC分类器
    print("微调MIC分类器...")
    mic_model = mic_classifier_model
    
    # 冻结除最后几层外的所有层
    for layer in mic_model.layers[:-2]:
        layer.trainable = False
    
    mic_model.compile(
        optimizer=Adam(lr=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    history_mic = mic_model.fit(
        train_seqs, train_mic,
        validation_data=(val_seqs, val_mic),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # 6. 保存微调后的模型
    print("保存微调后的模型...")
    os.makedirs('models/gram_finetuned', exist_ok=True)
    
    amp_model.save('models/gram_finetuned/amp_classifier_finetuned.h5')
    mic_model.save('models/gram_finetuned/mic_classifier_finetuned.h5')
    
    # 保存训练信息
    training_info = {
        'original_sequences_count': len(original_sequences),
        'filtered_sequences_count': len(filtered_sequences),
        'augmented_sequences_count': len(augmented_sequences),
        'training_sequences_count': len(train_seqs),
        'validation_sequences_count': len(val_seqs),
        'amp_final_loss': history_amp.history['loss'][-1],
        'amp_final_val_loss': history_amp.history['val_loss'][-1],
        'mic_final_loss': history_mic.history['loss'][-1],
        'mic_final_val_loss': history_mic.history['val_loss'][-1],
        'high_quality_sequences': filtered_original_seqs
    }
    
    import json
    with open('models/gram_finetuned/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    
    print("✅ 微调完成！")
    print(f"模型保存位置: models/gram_finetuned/")
    print(f"训练信息保存位置: models/gram_finetuned/training_info.json")
    
    return True

def add_sequence_noise(sequence, noise_prob=0.1):
    """
    为序列添加轻微噪声（氨基酸替换）
    """
    if len(sequence) == 0:
        return None
    
    # 将sequence转换为可修改的列表
    seq_one_hot = sequence.copy()
    seq_length = np.sum(seq_one_hot.sum(axis=1) > 0)  # 实际序列长度
    
    # 随机选择一个位置进行轻微修改
    if seq_length > 1 and np.random.random() < noise_prob:
        pos = np.random.randint(0, seq_length)
        # 随机选择一个新的氨基酸
        new_aa = np.random.randint(0, 20)
        seq_one_hot[pos] = 0  # 清零
        seq_one_hot[pos, new_aa] = 1  # 设置新氨基酸
    
    return seq_one_hot

def main():
    """
    主函数：使用Gram-.fasta数据进行模型微调
    """
    print("=== 使用Gram-.fasta数据微调HydrAMP模型 ===")
    
    # 1. 检查输入文件
    fasta_file = "Gram-.fasta"
    if not os.path.exists(fasta_file):
        print(f"❌ 找不到输入文件: {fasta_file}")
        return
    
    # 2. 检查预训练模型
    required_models = ['models/amp_classifier', 'models/mic_classifier']
    for model_path in required_models:
        if not os.path.exists(model_path):
            print(f"❌ 找不到预训练模型: {model_path}")
            print("请先运行: sh get_data.sh")
            return
    
    # 3. 准备数据
    print("步骤1: 准备Gram数据...")
    gram_sequences, gram_labels, original_sequences = prepare_gram_data(fasta_file)
    
    if len(gram_sequences) < 10:
        print("❌ 有效序列太少，无法进行训练")
        return
    
    # 4. 进行微调训练
    print("步骤2: 开始微调训练...")
    success = fine_tune_with_gram_data(gram_sequences, gram_labels, original_sequences)
    
    if success:
        print("\n🎉 微调训练完成！")
        print("\n下一步：")
        print("1. 使用微调后的模型生成数据:")
        print("   python generate_with_finetuned_model.py")
        print("2. 查看训练信息:")
        print("   cat models/gram_finetuned/training_info.json")
    else:
        print("\n❌ 微调训练失败")

if __name__ == "__main__":
    main()