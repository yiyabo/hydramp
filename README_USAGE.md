# HydrAMP 使用指南

## 概述

本项目提供了使用HydrAMP模型进行抗菌肽生成的完整流程，包括：
1. 使用预训练模型生成baseline数据
2. 使用自定义数据训练模型（可选）
3. 使用自定义模型生成数据

## 数据分析结果

### Gram-.fasta 文件分析
- **总序列数**: 768条
- **有效序列数**: 458条（≤25个氨基酸）
- **平均长度**: 30.40个氨基酸
- **长度范围**: 5-179个氨基酸
- **超过25个氨基酸的序列**: 310条（40.4%）

### 主要问题
1. **序列长度问题**: 40.4%的序列超过HydrAMP的最大长度限制（25个氨基酸）
2. **数据不完整**: 缺少负样本、MIC数据等训练必需数据
3. **样本数量**: 458条有效序列相对较少

## 使用方法

### 方案1：使用预训练模型生成baseline（推荐）

这是最简单且最可靠的方案：

```bash
# 1. 确保在服务器上已下载完整数据
sh get_data.sh

# 2. 运行baseline生成脚本
bash run_baseline_generation.sh

# 或者直接运行Python脚本
python generate_baseline.py
```

**输出文件**: `baseline_generated_300.csv`

### 方案2：使用自定义数据训练模型

⚠️ **警告**: 这个方案需要完整的训练数据集，包括：
- 负样本数据（非AMP序列）
- MIC数据（最小抑菌浓度）
- UniProt背景数据
- 预训练的分类器

```bash
# 1. 确保已下载完整数据
sh get_data.sh

# 2. 运行自定义训练
python train_with_custom_data.py

# 3. 使用训练好的模型生成数据
python generate_with_custom_model.py
```

## 文件说明

### 输入文件
- `Gram-.fasta`: 你的抗菌肽序列文件（768条序列）

### 输出文件
- `Gram-_filtered.csv`: 过滤后的有效序列（458条）
- `baseline_generated_300.csv`: 使用预训练模型生成的300条抗菌肽
- `custom_model_generated_300.csv`: 使用自定义模型生成的300条抗菌肽

### 脚本文件
- `generate_baseline.py`: 使用预训练模型生成baseline
- `train_with_custom_data.py`: 使用自定义数据训练模型
- `generate_with_custom_model.py`: 使用自定义模型生成数据
- `run_baseline_generation.sh`: 一键运行baseline生成

## 生成数据格式

生成的CSV文件包含以下列：
- `sequence`: 氨基酸序列
- `amp`: AMP活性预测概率（0-1）
- `mic`: MIC预测概率（0-1）
- `length`: 序列长度
- `charge`: 电荷
- `hydrophobicity`: 疏水性
- `hydrophobic_moment`: 疏水矩
- `isoelectric_point`: 等电点

## 推荐流程

1. **第一步**: 运行方案1生成baseline数据
   ```bash
   bash run_baseline_generation.sh
   ```

2. **第二步**: 分析生成的数据
   ```python
   import pandas as pd
   df = pd.read_csv('baseline_generated_300.csv')
   print(df.describe())
   ```

3. **第三步**: 如果需要，可以尝试方案2进行自定义训练

## 注意事项

1. **环境要求**: 确保TensorFlow 2.2.1和相关依赖已安装
2. **内存需求**: 模型训练需要大量内存，建议使用GPU
3. **数据完整性**: 方案2需要完整的训练数据集
4. **序列长度**: 输入序列必须≤25个氨基酸
5. **随机性**: 生成结果具有随机性，可通过设置seed保证重现性

## 故障排除

### 常见问题
1. **模型文件不存在**: 运行 `sh get_data.sh` 下载数据
2. **内存不足**: 减少batch_size或使用更大内存的机器
3. **CUDA错误**: 确保CUDA版本与TensorFlow兼容
4. **依赖包错误**: 运行 `pip install -e .` 安装所有依赖

### 检查清单
- [ ] 已下载完整数据和模型文件
- [ ] Python环境配置正确
- [ ] 输入FASTA文件存在
- [ ] 有足够的磁盘空间
- [ ] GPU内存充足（如果使用GPU）