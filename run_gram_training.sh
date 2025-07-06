#!/bin/bash

# 使用Gram-.fasta数据进行HydrAMP模型微调的完整流程

echo "=== 使用Gram-.fasta数据微调HydrAMP模型 ==="
echo "开始时间: $(date)"

# 1. 检查环境
echo "步骤1: 检查环境..."
python --version
echo "检查关键包..."
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import keras; print('Keras:', keras.__version__)"
python -c "import amp; print('HydrAMP包导入成功')"

# 2. 检查输入数据
echo "步骤2: 检查输入数据..."
if [ ! -f "Gram-.fasta" ]; then
    echo "❌ 找不到Gram-.fasta文件"
    exit 1
fi

# 统计FASTA文件信息
echo "Gram-.fasta文件信息:"
echo "总行数: $(wc -l < Gram-.fasta)"
echo "序列数: $(grep -c "^>" Gram-.fasta)"

# 3. 检查预训练模型
echo "步骤3: 检查预训练模型..."
if [ ! -d "models/amp_classifier" ]; then
    echo "❌ 找不到预训练AMP分类器"
    echo "请先运行: sh get_data.sh"
    exit 1
fi

if [ ! -d "models/mic_classifier" ]; then
    echo "❌ 找不到预训练MIC分类器"
    echo "请先运行: sh get_data.sh"
    exit 1
fi

if [ ! -d "models/HydrAMP/37" ]; then
    echo "❌ 找不到预训练HydrAMP模型"
    echo "请先运行: sh get_data.sh"
    exit 1
fi

echo "✅ 所有预训练模型都存在"

# 4. 开始微调训练
echo "步骤4: 开始微调训练..."
echo "这可能需要10-30分钟，取决于您的硬件配置..."

python train_with_gram_data.py

# 检查训练结果
if [ -f "models/gram_finetuned/amp_classifier_finetuned.h5" ]; then
    echo "✅ AMP分类器微调成功"
else
    echo "❌ AMP分类器微调失败"
    exit 1
fi

if [ -f "models/gram_finetuned/mic_classifier_finetuned.h5" ]; then
    echo "✅ MIC分类器微调成功"
else
    echo "❌ MIC分类器微调失败"
    exit 1
fi

# 5. 显示训练信息
echo "步骤5: 显示训练信息..."
if [ -f "models/gram_finetuned/training_info.json" ]; then
    echo "训练信息:"
    cat models/gram_finetuned/training_info.json
fi

echo "微调后的模型文件:"
ls -la models/gram_finetuned/

# 6. 使用微调模型生成数据
echo "步骤6: 使用微调模型生成300条抗菌肽..."
python generate_with_finetuned_model.py

# 7. 检查生成结果
if ls gram_finetuned_generated_*.csv 1> /dev/null 2>&1; then
    RESULT_FILE=$(ls gram_finetuned_generated_*.csv | head -n1)
    echo "✅ 生成成功！"
    echo "输出文件: $RESULT_FILE"
    echo "文件大小: $(du -h "$RESULT_FILE")"
    
    # 统计生成的序列数
    if command -v wc >/dev/null 2>&1; then
        LINE_COUNT=$(($(wc -l < "$RESULT_FILE") - 1))
        echo "生成序列数: $LINE_COUNT"
    fi
    
    # 显示前几行
    echo "前5行预览:"
    head -6 "$RESULT_FILE"
    
else
    echo "❌ 生成失败，未找到输出文件"
    exit 1
fi

echo ""
echo "完成时间: $(date)"
echo "=== Gram数据微调和生成完成 ==="
echo ""
echo "📁 输出文件说明:"
echo "- models/gram_finetuned/: 微调后的模型文件"
echo "- models/gram_finetuned/training_info.json: 训练过程信息"
echo "- gram_finetuned_generated_*.csv: 基于你的Gram数据生成的抗菌肽"
echo ""
echo "🎉 现在你有了基于自己数据训练的抗菌肽生成模型！"