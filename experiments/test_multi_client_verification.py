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
import concurrent.futures
import threading
import os


def get_verification_time_costs_with_multi_clients():
    print("===============time costs with different clients======================")

    # load signature keys, cht_keys_params, HE_keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)
    HE = load_HE_keys()

    all_models_data = {}
    model_id_mapping = {}
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/model_id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        encrypted_model_param = extract_data_from_hash_node(encrypted_path)
        for name, param in encrypted_model_param.items():
            all_models_data[model_id][name] = param

            # load CHT
    CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_10.tree")
    print(f"CHT load successfully {CHT}")

    # simulate the cloud server
    cloud_server = ModelCloudServer(HE, CHT, all_models_data)

    # 指定要验证的模型ID
    specify_model_id = "lenet1"

    # 获取模型验证包 - 只获取一次，以便所有客户端使用相同的验证包
    model_package = cloud_server.get_model(specify_model_id)

    # 创建CSV文件并写入表头
    with open(f"../table/multi_client_verification_time_costs_threaded.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['client_count', 'avg_time'])


    # 测试不同数量的客户端
    client_counts = [20, 40, 60, 80, 100]
    results = {}

    # 计算最大并发线程数，避免创建过多线程
    max_threads = min(100, max(client_counts))

    # 定义一个线程安全的列表来存储验证时间
    verification_times_lock = threading.Lock()

    # 定义客户端验证函数
    def client_verify(client, model_pkg, verification_times):
        start_time = time.perf_counter()
        verification_result = client.verify_model(model_pkg)
        end_time = time.perf_counter()

        verification_time = (end_time - start_time) * 1000  # 转换为毫秒

        # 线程安全地添加验证时间
        with verification_times_lock:
            verification_times.append(verification_time)

        return verification_time

    for num_clients in client_counts:
        # 创建指定数量的客户端
        clients = []
        for i in range(num_clients):
            client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)
            clients.append(client)

            # 计算并发因子 - 根据客户端数量调整并发度
        concurrency_factor = min(num_clients, max_threads)

        verification_times = []
        total_start_time = time.perf_counter()
        completed_count = 0

        # 使用线程池执行客户端验证
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_factor) as executor:
            # 提交所有客户端验证任务
            future_to_client = {
                executor.submit(client_verify, client, model_package, verification_times): i
                for i, client in enumerate(clients)
            }

            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_client):
                client_idx = future_to_client[future]
                try:
                    verification_time = future.result()
                    completed_count += 1
                    # 每完成25%的客户端打印一次进度
                    if completed_count % max(1, num_clients // 4) == 0:
                        print(f"已完成 {completed_count}/{num_clients} 个客户端的验证")
                except Exception as e:
                    print(f"客户端 {client_idx} 验证时发生错误: {e}")
                    completed_count += 1

        total_end_time = time.perf_counter()
        total_time = (total_end_time - total_start_time) * 1000  # 转换为毫秒

        # 计算统计数据
        avg_time = sum(verification_times) / len(verification_times)
        min_time = min(verification_times)
        max_time = max(verification_times)

        # 保存结果
        results[num_clients] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_time': total_time,
            'concurrency_factor': concurrency_factor
        }

        # 创建CSV文件并写入表头
        with open(f"../table/multi_client_verification_time_costs_threaded.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([num_clients, min_time])

    # 打印详细结果
    print("\n==== 详细统计结果 ====")
    print("客户端数量\t平均时间(ms)\t最小时间(ms)\t最大时间(ms)\t总时间(ms)\t并发度")
    for num_clients, stats in results.items():
        print(
            f"{num_clients}\t\t{stats['avg_time']:.2f}\t\t{stats['min_time']:.2f}\t\t{stats['max_time']:.2f}\t\t{stats['total_time']:.2f}\t\t{stats['concurrency_factor']}")

    return results


def draw_time_vs_client_count_from_file(file_path):
    """
    从文件读取数据并绘制时间随客户端数量变化的柱状图

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
    required_columns = ['client_count', 'avg_time']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件缺少必要的列 '{col}'")
            return

            # 设置字体样式
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 获取客户端数量和时间数据
    client_counts = df['client_count'].tolist()
    avg_times = df['avg_time'].tolist()

    # 设置x轴位置
    x = np.arange(len(client_counts))
    width = 0.6  # 柱宽度

    # 绘制柱状图
    bars = plt.bar(x, avg_times, width, color='#1f77b4')

    # # 在柱上添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
    #              f'{height:.2f}',
    #              ha='center', va='bottom', fontsize=10)

        # 添加标签和标题
    plt.xlabel('Number of clients', fontsize=12, fontweight='bold')
    plt.ylabel('Average verification time cost (ms)', fontsize=12, fontweight='bold')
    # plt.title('Average Time vs. Number of Clients', fontsize=14, fontweight='bold')

    # 设置x轴标签
    plt.xticks(x, client_counts, fontsize=12)
    plt.yticks(fontsize=12)

    # # 添加网格线使图表更易读
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 创建输出目录(如果不存在)
    output_dir = '../figure'
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, '../figure/avg_time_vs_client_count.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存图表到: {output_path}")

    # 显示图表
    plt.show()

if __name__ == "__main__":
    # res = get_verification_time_costs_with_multi_clients()
    # print(res)

    draw_time_vs_client_count_from_file("../table/multi_client_verification_time_costs_threaded.csv")

