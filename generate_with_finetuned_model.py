#!/usr/bin/env python3
"""
使用Gram数据微调后的模型生成300条抗菌肽
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from keras.models import load_model

# 导入HydrAMP模块
from amp.config import LATENT_DIM
from amp.data_utils import sequence as du_sequence
from amp.utils.basic_model_serializer import load_master_model_components
from amp.utils.generate_peptides import translate_peptide
from amp.utils.phys_chem_propterties import calculate_physchem_prop
from amp.utils.seed import set_seed
import joblib

def generate_with_finetuned_classifiers(n_target=300, seed=42):
    """
    使用微调后的分类器生成抗菌肽
    """
    print("=== 使用Gram微调模型生成抗菌肽 ===")
    
    # 1. 检查微调模型是否存在
    finetuned_amp_path = 'models/gram_finetuned/amp_classifier_finetuned.h5'
    finetuned_mic_path = 'models/gram_finetuned/mic_classifier_finetuned.h5'
    
    if not os.path.exists(finetuned_amp_path):
        print(f"❌ 找不到微调的AMP分类器: {finetuned_amp_path}")
        print("请先运行: python train_with_gram_data.py")
        return None
    
    if not os.path.exists(finetuned_mic_path):
        print(f"❌ 找不到微调的MIC分类器: {finetuned_mic_path}")
        print("请先运行: python train_with_gram_data.py")
        return None
    
    # 2. 加载模型组件
    print("加载模型组件...")
    
    # 加载原始的编码器和解码器
    model_path = "models/HydrAMP/37"
    decomposer_path = "models/HydrAMP/pca_decomposer.joblib"
    
    components = load_master_model_components(model_path, return_master=True, softmax=False)
    encoder, decoder, _, _, master = components
    
    # 加载微调后的分类器
    print("加载微调后的分类器...")
    finetuned_amp_classifier = load_model(finetuned_amp_path)
    finetuned_mic_classifier = load_model(finetuned_mic_path)
    
    # 加载PCA分解器
    latent_decomposer = joblib.load(decomposer_path)
    
    # 3. 读取训练信息
    training_info_path = 'models/gram_finetuned/training_info.json'
    if os.path.exists(training_info_path):
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        print(f"训练信息: 使用了{training_info['original_sequences_count']}条原始Gram序列")
        print(f"高质量序列: {training_info['filtered_sequences_count']}条")
        print(f"数据增强后: {training_info['augmented_sequences_count']}条")
    
    # 4. 生成新序列
    print(f"开始生成{n_target}条序列...")
    set_seed(seed)
    
    accepted_sequences = []
    accepted_amp = []
    accepted_mic = []
    
    batch_size = 100
    attempts_per_z = 64
    max_iterations = 20  # 防止无限循环
    iteration = 0
    
    while len(accepted_sequences) < n_target and iteration < max_iterations:
        iteration += 1
        print(f"迭代 {iteration}: 当前已生成 {len(accepted_sequences)}/{n_target} 条序列")
        
        # 生成潜在向量
        current_batch = min(batch_size, n_target - len(accepted_sequences))
        z = np.random.normal(size=(current_batch, LATENT_DIM))
        z = latent_decomposer.inverse_transform(z)
        z = np.vstack([z] * attempts_per_z)
        
        # 添加条件（AMP=1, MIC=1）
        c_amp = np.ones((z.shape[0], 1))
        c_mic = np.ones((z.shape[0], 1))
        z_cond = np.hstack([z, c_amp, c_mic])
        
        # 解码生成序列
        candidate = decoder.predict(z_cond, verbose=0, batch_size=1000)
        candidate_index_decoded = candidate.argmax(axis=2)
        generated_sequences = [translate_peptide(pep) for pep in candidate_index_decoded]
        
        # 使用微调后的分类器预测
        generated_amp = finetuned_amp_classifier.predict(candidate_index_decoded, verbose=0).flatten()
        generated_mic = finetuned_mic_classifier.predict(candidate_index_decoded, verbose=0).flatten()
        
        # 选择最佳序列
        generated_sequences = np.array(generated_sequences).reshape(attempts_per_z, -1)
        generated_amp = generated_amp.reshape(attempts_per_z, -1)
        generated_mic = generated_mic.reshape(attempts_per_z, -1)
        
        # 对每个原始z向量选择最佳序列
        best_indices = generated_amp.argmax(axis=0)
        num_z_vectors = generated_sequences.shape[1]
        
        for i in range(num_z_vectors):
            best_idx = best_indices[i]
            seq = generated_sequences[best_idx, i]
            amp_prob = generated_amp[best_idx, i]
            mic_prob = generated_mic[best_idx, i]
            
            # 过滤条件（相比原版更严格，因为使用了微调模型）
            if (amp_prob > 0.85 and  # 提高AMP阈值
                mic_prob > 0.3 and   # MIC阈值
                5 <= len(seq) <= 25 and 
                seq not in accepted_sequences and
                not has_problematic_patterns(seq)):
                
                accepted_sequences.append(seq)
                accepted_amp.append(amp_prob)
                accepted_mic.append(mic_prob)
                
                if len(accepted_sequences) >= n_target:
                    break
    
    if len(accepted_sequences) < n_target:
        print(f"⚠️  只生成了{len(accepted_sequences)}条序列（目标{n_target}条）")
        print("这可能是因为微调模型的过滤条件较严格")
    
    # 5. 计算物理化学性质并保存
    if len(accepted_sequences) > 0:
        print("计算物理化学性质...")
        physchem_props = calculate_physchem_prop(accepted_sequences)
        
        # 创建结果DataFrame
        result_data = {
            'sequence': accepted_sequences,
            'amp': accepted_amp,
            'mic': accepted_mic,
            'length': [len(seq) for seq in accepted_sequences],
            'model_type': ['gram_finetuned'] * len(accepted_sequences)  # 标记模型类型
        }
        
        # 添加物理化学性质
        result_data.update(physchem_props)
        
        df = pd.DataFrame(result_data)
        
        # 保存结果
        output_file = f"gram_finetuned_generated_{len(df)}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ 结果已保存到: {output_file}")
        
        # 显示统计信息
        print("\n=== 生成数据统计 ===")
        print(f"总序列数: {len(df)}")
        print(f"平均长度: {df['length'].mean():.2f}")
        print(f"长度范围: {df['length'].min()}-{df['length'].max()}")
        print(f"平均AMP概率: {df['amp'].mean():.4f}")
        print(f"平均MIC概率: {df['mic'].mean():.4f}")
        
        # 比较原始Gram数据
        if os.path.exists(training_info_path):
            print(f"\n=== 与原始Gram数据对比 ===")
            high_quality_seqs = training_info.get('high_quality_sequences', [])
            if high_quality_seqs:
                print(f"原始Gram高质量序列数: {len(high_quality_seqs)}")
                print(f"生成序列数: {len(df)}")
                print(f"扩展倍数: {len(df)/len(high_quality_seqs):.1f}x")
        
        print("\n前5个生成的序列:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            print(f"{i+1}. {row['sequence']} (长度:{row['length']}, AMP:{row['amp']:.4f})")
        
        return df
    else:
        print("❌ 没有生成任何符合条件的序列")
        return None

def has_problematic_patterns(seq):
    """检查序列是否有问题模式"""
    # 检查连续的疏水氨基酸
    hydrophobic = 'AILMFPWVY'
    consecutive_hydrophobic = 0
    max_consecutive_hydrophobic = 0
    
    for i, aa in enumerate(seq):
        if aa in hydrophobic:
            consecutive_hydrophobic += 1
            max_consecutive_hydrophobic = max(max_consecutive_hydrophobic, consecutive_hydrophobic)
        else:
            consecutive_hydrophobic = 0
        
        # 检查5个氨基酸窗口内的重复
        if i >= 4:
            window = seq[i-4:i+1]
            unique_count = len(set(window))
            if unique_count <= 2:
                return True
    
    # 连续疏水氨基酸过多
    if max_consecutive_hydrophobic > 3:
        return True
    
    # 检查半胱氨酸
    if 'C' in seq:
        return True
    
    return False

def main():
    """主函数"""
    print("=== 使用Gram微调模型生成抗菌肽 ===")
    
    # 检查是否已完成微调训练
    if not os.path.exists('models/gram_finetuned/amp_classifier_finetuned.h5'):
        print("❌ 未找到微调后的模型")
        print("请先运行: python train_with_gram_data.py")
        return
    
    # 生成数据
    df = generate_with_finetuned_classifiers(n_target=300, seed=42)
    
    if df is not None:
        print(f"\n🎉 成功使用Gram微调模型生成{len(df)}条抗菌肽！")
        print("这些序列是基于你的Gram-.fasta数据训练的模型生成的。")
    else:
        print("\n❌ 生成失败")

if __name__ == "__main__":
    main()