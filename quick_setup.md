# HydrAMP Python 3.8 快速设置指南

## 方法1: 使用Conda（推荐）

```bash
# 1. 创建Python 3.8环境
conda create -n hydramp_py38 python=3.8 -y

# 2. 激活环境
conda activate hydramp_py38

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装HydrAMP包
pip install -e .

# 5. 下载数据和模型（如果还没下载）
sh get_data.sh

# 6. 运行baseline生成
python generate_baseline.py
```

## 方法2: 使用virtualenv

```bash
# 1. 创建虚拟环境（确保系统Python是3.8）
python3.8 -m venv hydramp_env

# 2. 激活环境
source hydramp_env/bin/activate  # Linux/Mac
# 或 hydramp_env\Scripts\activate  # Windows

# 3. 升级pip
pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装HydrAMP包
pip install -e .

# 6. 运行生成
python generate_baseline.py
```

## 自动化脚本

如果使用Conda，可以直接运行：

```bash
bash setup_environment.sh
```

## 验证安装

安装完成后验证：

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import amp; print('HydrAMP导入成功')"
python -c "from typing import Literal; print('Literal支持正常')"
```

## 常见问题

1. **如果biopython安装失败**：
   ```bash
   pip install biopython==1.78  # 尝试更低版本
   ```

2. **如果TensorFlow安装失败**：
   ```bash
   pip install tensorflow==2.2.0  # 尝试稍低版本
   ```

3. **如果GPU版本问题**：
   ```bash
   pip install tensorflow-gpu==2.2.1  # 如果需要GPU支持
   ```

## 下一步

环境设置成功后：

1. 确保数据已下载：`sh get_data.sh`
2. 运行baseline生成：`python generate_baseline.py`
3. 查看结果：`baseline_generated_300.csv`