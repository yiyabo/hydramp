#!/usr/bin/env python3
"""
使用自定义数据（Gram-.fasta）训练HydrAMP模型的完整流程
注意：这个脚本需要完整的训练数据集才能正常运行
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
import amp.data_utils.data_loader as data_loader
import amp.data_utils.sequence as du_sequence
from amp.config import hydra, MIN_LENGTH, MAX_LENGTH, LATENT_DIM, MIN_KL, RCL_WEIGHT, HIDDEN_DIM, MAX_TEMPERATURE
from amp.models.decoders import amp_expanded_decoder
from amp.models.encoders import amp_expanded_encoder
from amp.models.master import master
from amp.utils import basic_model_serializer, callback, generator
from keras import backend, layers
from keras.optimizers import Adam

def prepare_custom_data(fasta_file, output_positive_csv):
    """
    准备自定义数据：将FASTA转换为训练所需的CSV格式
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
    
    # 过滤序列：长度<=25且只包含标准氨基酸
    filtered_data = []
    for seq_id, seq in zip(ids, sequences):
        if MIN_LENGTH <= len(seq) <= MAX_LENGTH and check_if_std_aa(seq):
            filtered_data.append({'Name': seq_id, 'Sequence': seq})
    
    # 保存为CSV
    df = pd.DataFrame(filtered_data)
    df.to_csv(output_positive_csv, index=False)
    
    print(f"已保存{len(filtered_data)}条有效序列到: {output_positive_csv}")
    return len(filtered_data)

def create_minimal_training_setup():
    """
    创建最小化训练配置
    警告：这个配置可能不会产生好的结果，因为缺少必要的训练数据
    """
    print("⚠️  警告：当前配置缺少以下必要数据：")
    print("   - 负样本数据（非AMP序列）")
    print("   - MIC数据（最小抑菌浓度）")
    print("   - UniProt背景数据")
    print("   - 预训练的AMP和MIC分类器")
    print("   这将严重影响模型性能！")
    
    # 检查必要文件
    required_files = [
        'data/unlabelled_negative.csv',  # 负样本
        'data/mic_data.csv',             # MIC数据
        'data/Uniprot_0_25_train.csv',   # UniProt训练数据
        'data/Uniprot_0_25_val.csv',     # UniProt验证数据
        'models/amp_classifier',         # AMP分类器
        'models/mic_classifier'          # MIC分类器
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ 缺少以下必要文件：")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n请先运行 'sh get_data.sh' 下载完整数据集")
        return False
    
    return True

def train_hydramp_with_custom_data(custom_positive_csv):
    """
    使用自定义数据训练HydrAMP模型
    """
    print("开始训练HydrAMP模型...")
    
    # 设置随机种子
    seed = 7
    np.random.seed(seed)
    
    # 设置训练参数
    kl_weight = backend.variable(MIN_KL, name="kl_weight")
    tau = backend.variable(MAX_TEMPERATURE, name="temperature")
    
    # 1. 加载数据
    print("正在加载数据...")
    
    # 自定义正样本数据
    custom_df = pd.read_csv(custom_positive_csv)
    custom_sequences = custom_df['Sequence'].tolist()
    custom_x = du_sequence.pad(du_sequence.to_one_hot(custom_sequences))
    custom_y = np.ones(len(custom_sequences))  # 全部标记为正样本
    
    # 原始训练数据（如果存在）
    try:
        data_manager = data_loader.AMPDataManager(
            'data/unlabelled_positive.csv',
            'data/unlabelled_negative.csv',
            min_len=MIN_LENGTH,
            max_len=MAX_LENGTH
        )
        amp_x, amp_y = data_manager.get_merged_data()
        
        # 合并数据
        combined_x = np.vstack([amp_x, custom_x])
        combined_y = np.concatenate([amp_y, custom_y])
        
    except Exception as e:
        print(f"无法加载原始数据，仅使用自定义数据: {e}")
        combined_x = custom_x
        combined_y = custom_y
    
    # 数据分割
    amp_x_train, amp_x_test, amp_y_train, amp_y_test = train_test_split(
        combined_x, combined_y, test_size=0.1, random_state=36
    )
    amp_x_train, amp_x_val, amp_y_train, amp_y_val = train_test_split(
        amp_x_train, amp_y_train, test_size=0.2, random_state=36
    )
    
    # 2. 加载MIC数据
    try:
        ecoli_df = pd.read_csv('data/mic_data.csv')
        mask = (ecoli_df['sequence'].str.len() <= MAX_LENGTH) & (ecoli_df['sequence'].str.len() >= MIN_LENGTH)
        ecoli_df = ecoli_df.loc[mask]
        mic_x = du_sequence.pad(du_sequence.to_one_hot(ecoli_df['sequence']))
        mic_y = ecoli_df.value.values
        
        mic_x_train, mic_x_test, mic_y_train, mic_y_test = train_test_split(
            mic_x, mic_y, test_size=0.1, random_state=36
        )
        mic_x_train, mic_x_val, mic_y_train, mic_y_val = train_test_split(
            mic_x_train, mic_y_train, test_size=0.2, random_state=36
        )
        
    except Exception as e:
        print(f"无法加载MIC数据: {e}")
        # 使用自定义数据的一部分作为MIC数据
        mic_x_train, mic_x_val = amp_x_train[:100], amp_x_val[:50]
        mic_y_train, mic_y_val = np.random.uniform(0, 2, 100), np.random.uniform(0, 2, 50)
    
    # 3. 加载UniProt数据
    try:
        uniprot_x_train = np.array(du_sequence.pad(du_sequence.to_one_hot(
            pd.read_csv('data/Uniprot_0_25_train.csv').Sequence
        )))
        uniprot_x_val = np.array(du_sequence.pad(du_sequence.to_one_hot(
            pd.read_csv('data/Uniprot_0_25_val.csv').Sequence
        )))
    except Exception as e:
        print(f"无法加载UniProt数据: {e}")
        # 使用自定义数据的一部分
        uniprot_x_train, uniprot_x_val = amp_x_train[:1000], amp_x_val[:200]
    
    # 4. 加载预训练分类器
    try:
        bms = basic_model_serializer.BasicModelSerializer()
        amp_classifier = bms.load_model('models/amp_classifier')
        mic_classifier = bms.load_model('models/mic_classifier')
        amp_classifier_model = amp_classifier()
        mic_classifier_model = mic_classifier()
        print("成功加载预训练分类器")
    except Exception as e:
        print(f"❌ 无法加载预训练分类器: {e}")
        print("训练无法继续，请确保分类器文件存在")
        return False
    
    # 5. 构建模型
    print("正在构建模型...")
    
    encoder = amp_expanded_encoder.AMPEncoderFactory.get_default(HIDDEN_DIM, LATENT_DIM, MAX_LENGTH)
    decoder = amp_expanded_decoder.AMPDecoderFactory.build_default(LATENT_DIM + 2, tau, MAX_LENGTH)
    
    master_model = master.MasterAMPTrainer(
        amp_classifier=amp_classifier,
        mic_classifier=mic_classifier,
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,
        rcl_weight=RCL_WEIGHT,
        master_optimizer=Adam(lr=1e-3),
        loss_weights=hydra,
    )
    
    # 构建Keras模型
    input_to_encoder = layers.Input(shape=(MAX_LENGTH,))
    encoder_model = encoder(input_to_encoder)
    input_to_decoder = layers.Input(shape=(LATENT_DIM + 2,))
    decoder_model = decoder(input_to_decoder)
    
    master_keras_model = master_model.build(input_shape=(MAX_LENGTH, 21))
    
    # 6. 准备训练数据
    print("正在准备训练数据...")
    
    # 预测分类器输出
    amp_amp_train = amp_classifier_model.predict(amp_x_train, verbose=1, batch_size=10000).reshape(len(amp_x_train))
    amp_mic_train = mic_classifier_model.predict(amp_x_train, verbose=1, batch_size=10000).reshape(len(amp_x_train))
    amp_amp_val = amp_classifier_model.predict(amp_x_val, verbose=1, batch_size=10000).reshape(len(amp_x_val))
    amp_mic_val = mic_classifier_model.predict(amp_x_val, verbose=1, batch_size=10000).reshape(len(amp_x_val))
    
    mic_amp_train = amp_classifier_model.predict(mic_x_train, verbose=1, batch_size=10000).reshape(len(mic_x_train))
    mic_mic_train = mic_classifier_model.predict(mic_x_train, verbose=1, batch_size=10000).reshape(len(mic_x_train))
    mic_amp_val = amp_classifier_model.predict(mic_x_val, verbose=1, batch_size=10000).reshape(len(mic_x_val))
    mic_mic_val = mic_classifier_model.predict(mic_x_val, verbose=1, batch_size=10000).reshape(len(mic_x_val))
    
    uniprot_amp_train = amp_classifier_model.predict(uniprot_x_train, verbose=1, batch_size=10000).reshape(len(uniprot_x_train))
    uniprot_mic_train = mic_classifier_model.predict(uniprot_x_train, verbose=1, batch_size=10000).reshape(len(uniprot_x_train))
    uniprot_amp_val = amp_classifier_model.predict(uniprot_x_val, verbose=1, batch_size=10000).reshape(len(uniprot_x_val))
    uniprot_mic_val = mic_classifier_model.predict(uniprot_x_val, verbose=1, batch_size=10000).reshape(len(uniprot_x_val))
    
    # 7. 设置回调函数
    vae_callback = callback.VAECallback(
        encoder=encoder_model,
        decoder=decoder_model,
        tau=tau,
        kl_weight=kl_weight,
        amp_classifier=amp_classifier_model,
        mic_classifier=mic_classifier_model,
    )
    
    sm_callback = callback.SaveModelCallback(
        model=master_model,
        model_save_path="models/custom_trained/",
        name="HydrAMP_Custom"
    )
    
    # 8. 创建数据生成器
    training_generator = generator.concatenated_generator(
        uniprot_x_train, uniprot_amp_train, uniprot_mic_train,
        amp_x_train, amp_amp_train, amp_mic_train,
        mic_x_train, mic_amp_train, mic_mic_train,
        128
    )
    
    validation_generator = generator.concatenated_generator(
        uniprot_x_val, uniprot_amp_val, uniprot_mic_val,
        amp_x_val, amp_amp_val, amp_mic_val,
        mic_x_val, mic_amp_val, mic_mic_val,
        128
    )
    
    # 9. 开始训练
    print("开始训练...")
    
    # 计算steps
    total_train_samples = len(uniprot_x_train) + len(amp_x_train) + len(mic_x_train)
    total_val_samples = len(uniprot_x_val) + len(amp_x_val) + len(mic_x_val)
    
    steps_per_epoch = max(1, total_train_samples // 128)
    validation_steps = max(1, total_val_samples // 128)
    
    print(f"训练样本数: {total_train_samples}")
    print(f"验证样本数: {total_val_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # 确保输出目录存在
    os.makedirs("models/custom_trained", exist_ok=True)
    
    try:
        history = master_keras_model.fit_generator(
            training_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,  # 减少epoch数量进行测试
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[vae_callback, sm_callback],
        )
        
        print("✅ 训练完成！")
        print(f"模型已保存到: models/custom_trained/")
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数：处理自定义数据并训练模型
    """
    print("=== HydrAMP 自定义数据训练脚本 ===")
    
    # 1. 准备自定义数据
    fasta_file = "Gram-.fasta"
    custom_positive_csv = "custom_positive.csv"
    
    if not os.path.exists(fasta_file):
        print(f"❌ 找不到FASTA文件: {fasta_file}")
        return
    
    # 转换FASTA为CSV
    num_sequences = prepare_custom_data(fasta_file, custom_positive_csv)
    
    if num_sequences < 10:
        print("❌ 有效序列数量太少，无法进行训练")
        return
    
    # 2. 检查训练环境
    if not create_minimal_training_setup():
        print("❌ 训练环境不完整")
        return
    
    # 3. 开始训练
    success = train_hydramp_with_custom_data(custom_positive_csv)
    
    if success:
        print("\n🎉 训练完成！")
        print("现在可以使用训练好的模型进行生成:")
        print("python generate_with_custom_model.py")
    else:
        print("\n❌ 训练失败")

if __name__ == "__main__":
    main()