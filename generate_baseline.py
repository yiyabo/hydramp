#!/usr/bin/env python3
"""
使用预训练的HydrAMP模型生成300条抗菌肽作为baseline
"""

import os
import sys
import pandas as pd
import numpy as np
from amp.inference import HydrAMPGenerator
from amp.utils.seed import set_seed

def generate_baseline_data(model_path, decomposer_path, output_file, n_target=300, seed=42):
    """
    使用预训练模型生成baseline数据
    
    Args:
        model_path: 预训练模型路径
        decomposer_path: PCA分解器路径
        output_file: 输出CSV文件路径
        n_target: 目标生成数量
        seed: 随机种子
    """
    print(f"正在加载预训练模型...")
    print(f"模型路径: {model_path}")
    print(f"分解器路径: {decomposer_path}")
    
    # 初始化生成器
    generator = HydrAMPGenerator(model_path, decomposer_path)
    
    print(f"开始生成{n_target}条抗菌肽序列...")
    
    # 无约束生成抗菌肽
    generated_peptides = generator.unconstrained_generation(
        mode="amp",
        n_target=n_target,
        seed=seed,
        filter_out=True,
        properties=True,
        n_attempts=64,
        filter_hydrophobic_clusters=True,
        filter_repetitive_clusters=True,
        filter_cysteins=True,
        filter_known_amps=True
    )
    
    print(f"成功生成{len(generated_peptides)}条序列")
    
    # 转换为DataFrame
    df = pd.DataFrame(generated_peptides)
    
    # 重新排序列
    columns_order = ['sequence', 'amp', 'mic', 'length', 'charge', 
                    'hydrophobicity', 'hydrophobic_moment', 'isoelectric_point']
    df = df[columns_order]
    
    # 保存到CSV
    df.to_csv(output_file, index=False)
    
    print(f"结果已保存到: {output_file}")
    
    # 显示统计信息
    print("\n=== 生成数据统计 ===")
    print(f"总序列数: {len(df)}")
    print(f"平均长度: {df['length'].mean():.2f}")
    print(f"长度范围: {df['length'].min()}-{df['length'].max()}")
    print(f"平均AMP概率: {df['amp'].mean():.4f}")
    print(f"平均MIC概率: {df['mic'].mean():.4f}")
    print(f"平均电荷: {df['charge'].mean():.2f}")
    print(f"平均疏水性: {df['hydrophobicity'].mean():.4f}")
    
    print("\n前5个生成的序列:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"{i+1}. {row['sequence']} (长度:{row['length']}, AMP:{row['amp']:.4f})")
    
    return df

def main():
    # 设置路径 - 根据服务器上的实际路径调整
    model_path = "models/HydrAMP/37"  # 最新训练的模型
    decomposer_path = "models/HydrAMP/pca_decomposer.joblib"
    output_file = "baseline_generated_300.csv"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请确保已下载模型文件，或调整路径")
        return
    
    if not os.path.exists(decomposer_path):
        print(f"错误: 分解器路径不存在: {decomposer_path}")
        print("请确保已下载分解器文件，或调整路径")
        return
    
    try:
        # 生成baseline数据
        df = generate_baseline_data(
            model_path=model_path,
            decomposer_path=decomposer_path,
            output_file=output_file,
            n_target=300,
            seed=42
        )
        
        print(f"\n✅ 成功生成300条baseline抗菌肽数据!")
        
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()