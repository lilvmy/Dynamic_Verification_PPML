from initialization.setup import load_ecdsa_keys, load_HE_keys
from dual_verification_tree.CHT_utils import load_cht_keys
from level_homomorphic_encryption.encrypted_process_model import extract_data_from_hash_node
from dual_verification_tree.build_CHT import load_chameleon_hash_tree
from simulator_client_cloud.model_verification_demo import ModelCloudServer, ModelVerifier
import csv
import time
from utils.util import get_size
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback


def get_time_storage_costs_with_cloud_tamper():
    print("===============model verification, the cloud server try to tamper model parameters======================")

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

            # 使用所有可用模型
    available_models = list(model_id_mapping.keys())
    model_count = len(available_models)
    print(f"总共有 {model_count} 个可用模型: {available_models}")

    # 显示每个模型的参数数量
    for model_id in available_models:
        param_count = len(all_models_data[model_id])
        print(f"模型 {model_id} 有 {param_count} 个参数")

        # Create CSV file with header
    with open(f"../table/time_storage_costs_with_cloud_tamper_client_specify_model.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_id',
            'tampered_params_count',
            'total_time',
            'storage_costs',
        ])

        # 对每个模型进行渐进式篡改测试
    final_packages = None
    for model_index, model_id in enumerate(available_models, 1):
        print(f"\n====== [{model_index}/{model_count}] 对模型 {model_id} 进行渐进式篡改测试 ======")

        # 获取模型参数总数
        total_params = len(all_models_data[model_id])
        print(f"模型 {model_id} 共有 {total_params} 个参数")

        # 逐步增加篡改参数数量，从1到所有参数
        for tamper_count in range(1, total_params + 1):
            print(f"\n--- 篡改模型 {model_id} 的前 {tamper_count}/{total_params} 个参数 ({tamper_count / total_params:.1%}) ---")

            try:
                # 每次测试都重新加载树和创建服务器实例，确保从原始状态开始
                CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_10.tree")
                print(f"CHT load successfully {CHT}")
                cloud_server = ModelCloudServer(HE, CHT, all_models_data)
                client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)

                # # 客户端注册所有模型参数
                # for mid, params in all_models_data.items():
                #     client.register_model(mid, params)

                # 测量获取篡改模型的时间
                start_time = time.time()
                # 每次都篡改从参数1到当前tamper_count的所有参数
                tampered_package = cloud_server.get_model(model_id, tamper_param_size=tamper_count, honest=False)
                end_time = time.time()
                get_model_time = (end_time - start_time) * 1000

                # 测量验证时间
                start_time = time.time()
                results = client.verify_model(tampered_package)
                end_time = time.time()
                verification_time = (end_time - start_time) * 1000

                # 计算总时间和存储成本
                total_time = get_model_time + verification_time
                storage_cost = get_size(tampered_package) / (1024 * 1024)

                # 计算失败的参数数量
                failed_params = sum(1 for r in results['params'].values() if not r['valid'])

                # 验证结果
                verification_result = '验证成功' if results['overall']['valid'] else '验证失败'

                # 输出验证结果
                print(f"验证结果:")
                print(f"  验证签名: {'成功' if results['signature']['valid'] else '失败'}")
                print(f"  验证模型路径: {'成功' if results['model_path']['valid'] else '失败'}")
                print(f"  参数验证: {len(results['params']) - failed_params} 个成功, {failed_params} 个失败")
                print(f"  整体结果: {verification_result}")
                print(f"  获取模型时间: {get_model_time:.2f} ms")
                print(f"  验证时间: {verification_time:.2f} ms")
                print(f"  总时间: {total_time:.2f} ms")
                print(f"  存储成本: {storage_cost:.2f} MB")

                # 记录数据到CSV
                with open(f"../table/time_storage_costs_with_cloud_tamper_client_specify_model.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_id,
                        tamper_count,
                        total_time,
                        storage_cost
                    ])

                    # 保存最后一次测试的包用于返回
                final_packages = tampered_package

            except Exception as e:
                print(f"错误: 测试模型 {model_id} 篡改 {tamper_count} 个参数时出现异常: {str(e)}")
                print(f"将跳过这个测试点并继续")
                traceback.print_exc()
                continue

    print("\n====== 全部模型渐进式篡改测试完成 ======")
    return final_packages

def draw_time_storage_costs(filepath):
    df = pd.read_csv(filepath)

    # Filter to include only the first 6 entries per model
    models = df['model_id'].unique()
    filtered_data = pd.DataFrame()

    for model in models:
        model_data = df[df['model_id'] == model].head(6)  # Get first 6 rows for each model
        filtered_data = pd.concat([filtered_data, model_data])

    # Set width of bars
    bar_width = 0.15
    positions = np.arange(len(models))

    # Colors for different parameter counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # First plot: Time Costs by Model Name
    plt.figure(figsize=(10, 6))
    for i in range(6):  # For each of the 6 parameter counts
        time_data = []
        for model in models:
            model_rows = filtered_data[filtered_data['model_id'] == model]
            if i < len(model_rows):
                time_data.append(model_rows.iloc[i]['total_time'])
            else:
                time_data.append(0)

        plt.bar(positions + (i - 2.5) * bar_width, time_data, bar_width,
                label=f'The cloud tampered with {i + 1} params', color=colors[i])

    plt.ylabel('Verification time costs (ms)',fontsize=12, fontweight='bold')
    plt.xlabel('Model name',fontsize=12, fontweight='bold')
    # plt.title('Time Costs by Model')
    plt.xticks(positions, models)
    plt.legend(title='The number of tampered parameters of a model')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../figure/time_costs_with_cloud_tamper_client_specify_model.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Second plot: Storage Costs by Model Name
    plt.figure(figsize=(10, 6))
    for i in range(6):  # For each of the 6 parameter counts
        storage_data = []
        for model in models:
            model_rows = filtered_data[filtered_data['model_id'] == model]
            if i < len(model_rows):
                storage_data.append(model_rows.iloc[i]['storage_costs'])
            else:
                storage_data.append(0)

        plt.bar(positions + (i - 2.5) * bar_width, storage_data, bar_width,
                label=f'The cloud tampered with {i + 1} params', color=colors[i])

    plt.ylabel('Verification storage costs (MB)',fontsize=12, fontweight='bold')
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    # plt.title('Storage Costs by Model')
    plt.xticks(positions, models)
    plt.legend(title='The number of tampered parameters of a model')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../figure/storage_costs_with_cloud_tamper_client_specify_model.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    res = get_time_storage_costs_with_cloud_tamper()

    draw_time_storage_costs("../table/time_storage_costs_with_cloud_tamper_client_specify_model.csv")

