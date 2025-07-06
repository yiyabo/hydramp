#!/bin/bash

# HydrAMP Python 3.8 环境设置脚本

echo "=== HydrAMP Python 3.8 环境设置 ==="

# 1. 创建Python 3.8虚拟环境
echo "步骤1: 创建Python 3.8虚拟环境..."
conda create -n hydramp_py38 python=3.8 -y

# 2. 激活环境
echo "步骤2: 激活环境..."
source activate hydramp_py38
# 或者使用: conda activate hydramp_py38

# 3. 升级pip
echo "步骤3: 升级pip..."
pip install --upgrade pip

# 4. 安装requirements
echo "步骤4: 安装依赖包..."
pip install -r requirements.txt

# 5. 安装HydrAMP包
echo "步骤5: 安装HydrAMP包..."
pip install -e .

# 6. 验证安装
echo "步骤6: 验证安装..."
python -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__)"
python -c "import amp; print('HydrAMP导入成功')"

echo "✅ 环境设置完成！"
echo ""
echo "使用方法："
echo "1. 激活环境: conda activate hydramp_py38"
echo "2. 运行生成: python generate_baseline.py"
echo "3. 退出环境: conda deactivate"