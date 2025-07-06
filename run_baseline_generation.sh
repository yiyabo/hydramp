#!/bin/bash

# HydrAMP Baseline 生成脚本
# 使用预训练模型生成300条抗菌肽作为baseline

echo "=== HydrAMP Baseline 生成脚本 ==="
echo "开始时间: $(date)"

# 1. 检查环境
echo "检查Python环境..."
python --version

echo "检查必要的包..."
python -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__)"
python -c "import keras; print('Keras版本:', keras.__version__)"
python -c "import pandas as pd; print('Pandas版本:', pd.__version__)"
python -c "import numpy as np; print('NumPy版本:', np.__version__)"

# 2. 检查数据和模型
echo "检查数据文件..."
if [ ! -f "Gram-.fasta" ]; then
    echo "❌ 找不到Gram-.fasta文件"
    exit 1
fi

echo "检查模型文件..."
if [ ! -d "models/HydrAMP/37" ]; then
    echo "❌ 找不到预训练模型，请先运行: sh get_data.sh"
    exit 1
fi

if [ ! -f "models/HydrAMP/pca_decomposer.joblib" ]; then
    echo "❌ 找不到PCA分解器，请先运行: sh get_data.sh"
    exit 1
fi

# 3. 安装HydrAMP包
echo "安装HydrAMP包..."
pip install -e .

# 4. 运行baseline生成
echo "运行baseline生成..."
python generate_baseline.py

# 5. 检查结果
if [ -f "baseline_generated_300.csv" ]; then
    echo "✅ Baseline生成成功！"
    echo "输出文件: baseline_generated_300.csv"
    echo "文件大小: $(du -h baseline_generated_300.csv)"
    echo "序列数量: $(wc -l < baseline_generated_300.csv)"
else
    echo "❌ Baseline生成失败"
    exit 1
fi

echo "完成时间: $(date)"
echo "=== Baseline生成完成 ==="