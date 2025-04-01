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
from utils.util import get_size

def client_get_normal_model():
    print("===============model verification, the cloud server does not tamper model parameters======================")

    # load signature keys, cht_keys_params, HE_keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)
    HE = load_HE_keys()


    all_models_data  = {}
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
    client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)

    # client register all of param of the model to audit
    for model_id_str, params in all_models_data.items():
        client.register_model(model_id_str, params)

    # the cloud server does not tamper model parameters
    print("\n=======the client request model normally based model_id==================")

    # Create CSV file with header
    with open(f"../table/client_verification_time_storage_costs_untamper.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'time_costs', 'storage_costs'])

    for model_id_str in model_id_mapping.keys():
        start_time = time.time()
        client_request_model = model_id_str
        print(f"the client request {client_request_model} model")

        model_package = cloud_server.get_model(client_request_model)
        model_package_size = get_size(model_package) / (1024 * 1024)

        print("\nthe client starts run verification operation:")
        results = client.verify_model(model_package)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        print(f"   the time costs of verification： {total_time}s")
        print(f"  verify signature: {'success' if results['signature']['valid'] else 'failure'}")
        print(f"  verify model path: {'success' if results['model_path']['valid'] else 'failure'}")
        print(f"  verify model parameters: {'all success ' if all(v['valid'] for v in results['params'].values()) else 'partial failure '}")
        print(f"  overall results: {'verification success' if results['overall']['valid'] else 'verification failure'}")

        with open(f"../table/client_verification_time_storage_costs_untamper.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_id_str, total_time, model_package_size])

        print(f"record experiment data: model name={model_id_str}, total time={total_time}ms, model verification package size={model_package_size}MB")


    return all_models_data

def draw_time_storage_costs(filepath):
    # set style for better visualization
    # sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Read the CSV file
    df = pd.read_csv(filepath)


    # figure 1: time costs vs model name
    plt.figure(figsize=(7, 5))
    time_bars = plt.bar(df['model_name'], df['time_costs'], color='lightcoral', width=0.6)
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    plt.ylabel('Verification time costs (ms)', fontsize=12, fontweight='bold')
    # plt.title('Time Costs vs Model Size', fontsize=14, fontweight='bold')
    plt.xticks(df['model_name'])
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # Add values on top of the bars
    # for bar in time_bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
    #              f'{height:.1f}',
    #              ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('../figure/client_verification_time_costs_untamper.png', dpi=300, bbox_inches='tight')
    plt.show()

    # figure 2: storage costs vs model size
    plt.figure(figsize=(7, 5))
    storage_bars = plt.bar(df['model_name'], df['storage_costs'], color='olivedrab', width=0.6)
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    plt.ylabel('Verification storage costs (MB)', fontsize=12, fontweight='bold')
    # plt.title('Storage Costs vs Model Size', fontsize=14, fontweight='bold')
    plt.xticks(df['model_name'])
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # Add values on top of the bars
    # for bar in storage_bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
    #              f'{height:.1f}',
    #              ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('../figure/client_verification_storage_costs_untamper.png', dpi=300, bbox_inches='tight')
    plt.show()  # Display the second figure (optional)


if __name__ == "__main__":
    # client_get_normal_model()

    draw_time_storage_costs("../table/client_verification_time_storage_costs_untamper.csv")
