#!/usr/bin/env python3
"""
使用自定义训练的模型生成抗菌肽
"""

import os
import sys
import pandas as pd
import numpy as np
from amp.inference import HydrAMPGenerator
from amp.utils.seed import set_seed

def generate_with_custom_model(model_path, decomposer_path, output_file, n_target=300, seed=42):
    """
    使用自定义训练的模型生成抗菌肽
    """
    print(f"正在加载自定义训练模型...")
    print(f"模型路径: {model_path}")
    print(f"分解器路径: {decomposer_path}")
    
    # 初始化生成器
    generator = HydrAMPGenerator(model_path, decomposer_path)
    
    print(f"开始生成{n_target}条抗菌肽序列...")
    
    # 无约束生成
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
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(generated_peptides)
    columns_order = ['sequence', 'amp', 'mic', 'length', 'charge', 
                    'hydrophobicity', 'hydrophobic_moment', 'isoelectric_point']
    df = df[columns_order]
    df.to_csv(output_file, index=False)
    
    print(f"✅ 成功生成{len(df)}条序列，保存到: {output_file}")
    return df

def main():
    # 自定义模型路径
    model_path = "models/custom_trained/HydrAMP_Custom/final_epoch"  # 最终训练epoch
    decomposer_path = "models/HydrAMP/pca_decomposer.joblib"  # 使用原始分解器
    output_file = "custom_model_generated_300.csv"
    
    if not os.path.exists(model_path):
        print(f"❌ 自定义模型不存在: {model_path}")
        print("请先运行 train_with_custom_data.py 训练模型")
        return
    
    try:
        df = generate_with_custom_model(
            model_path=model_path,
            decomposer_path=decomposer_path,
            output_file=output_file,
            n_target=300,
            seed=42
        )
        
        print("🎉 生成完成！")
        
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()