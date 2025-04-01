from initialization.setup import load_ecdsa_keys, load_HE_keys
from dual_verification_tree.CHT_utils import load_cht_keys
from level_homomorphic_encryption.encrypted_process_model import extract_data_from_hash_node
from dual_verification_tree.build_CHT import load_chameleon_hash_tree
from simulator_client_cloud.model_verification_demo import ModelCloudServer, ModelVerifier
import csv
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_verification_costs_with_different_schemes(file_path):
    """
    从CSV文件读取数据并绘制不同方案下模型的时间与存储成本柱状图

    参数:
    file_path -- CSV文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return

    # 从文件读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件出错: {str(e)}")
        return

    # 检查必要的列是否存在
    required_columns = ['schemes', 'model_name', 'time_costs', 'storage_costs']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件缺少必要的列 '{col}'")
            return

    # 转换数值列为浮点数
    df['time_costs'] = pd.to_numeric(df['time_costs'], errors='coerce')
    df['storage_costs'] = pd.to_numeric(df['storage_costs'], errors='coerce')

    # 设置字体
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 获取唯一的方案和模型名称
    schemes = sorted(df['schemes'].unique())
    model_names = df['model_name'].unique()

    # 为每个方案设置不同颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 不同颜色
    width = 0.25  # 条形宽度

    # 创建x轴位置
    x = np.arange(len(model_names))

    # ====== 时间成本图表 ======
    plt.figure(figsize=(7, 5))

    # 绘制时间柱状图
    for i, scheme in enumerate(schemes):
        scheme_data = df[df['schemes'] == scheme]

        # 确保数据按模型名称排序
        scheme_data = pd.merge(
            pd.DataFrame({'model_name': model_names}),
            scheme_data,
            on='model_name',
            how='left'
        ).fillna(0)

        plt.bar(x + (i - len(schemes) / 2 + 0.5) * width, scheme_data['time_costs'],
                width, label=scheme, color=colors[i % len(colors)])

    # 添加标签和标题
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    plt.ylabel('Verification time costs (ms)', fontsize=12, fontweight='bold')
    # plt.title('Time Costs for Different Schemes and Models', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, fontsize=10)
    plt.yticks(fontsize=10)

    # 创建对数刻度以更好地显示小值
    plt.yscale('log')

    # 格式化y轴刻度以提高可读性
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    # 添加图例
    plt.legend(fontsize=10)

    # 使用紧凑布局
    plt.tight_layout()

    # 创建输出目录(如果不存在)
    output_dir = '../figure'
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    time_output_path = os.path.join(output_dir, '../figure/verification_time_costs_with_different_models_and_schemes.png')
    plt.savefig(time_output_path, dpi=300, bbox_inches='tight')
    print(f"已保存时间成本图表到: {time_output_path}")

    # 显示图像
    plt.show()

    # ====== 存储成本图表 ======
    plt.figure(figsize=(7, 5))

    # 绘制存储柱状图
    for i, scheme in enumerate(schemes):
        scheme_data = df[df['schemes'] == scheme]

        # 确保数据按模型名称排序
        scheme_data = pd.merge(
            pd.DataFrame({'model_name': model_names}),
            scheme_data,
            on='model_name',
            how='left'
        ).fillna(0)

        plt.bar(x + (i - len(schemes) / 2 + 0.5) * width, scheme_data['storage_costs'],
                width, label=scheme, color=colors[i % len(colors)])

    # 添加标签和标题
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    plt.ylabel('Verification storage costs (MB)', fontsize=12, fontweight='bold')
    # plt.title('Storage Costs for Different Schemes and Models', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, fontsize=10)
    plt.yticks(fontsize=10)

    # 创建对数刻度以更好地显示小值
    plt.yscale('log')

    # 格式化y轴刻度以提高可读性
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.1f}'.format(x)))

    # 添加图例
    plt.legend(fontsize=10)

    # 使用紧凑布局
    plt.tight_layout()

    # 保存图像
    storage_output_path = os.path.join(output_dir, '../figure/verification_storage_costs_with_different_models_and_schemes.png')
    plt.savefig(storage_output_path, dpi=300, bbox_inches='tight')
    print(f"已保存存储成本图表到: {storage_output_path}")

    # 显示图像
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 指定CSV文件的路径
    file_path = '../table/verification_time_storage_costs_comparison.txt'
    draw_verification_costs_with_different_schemes(file_path)

