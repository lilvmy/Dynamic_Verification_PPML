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

    # Use all available models
    available_models = list(model_id_mapping.keys())
    model_count = len(available_models)
    print(f"Total {model_count} available models: {available_models}")

    # Display parameter count for each model
    for model_id in available_models:
        param_count = len(all_models_data[model_id])
        print(f"Model {model_id} has {param_count} parameters")

    # Create CSV file with header
    with open(f"../table/time_storage_costs_with_cloud_tamper_client_specify_model.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_id',
            'tampered_params_count',
            'total_time',
            'storage_costs',
        ])

    # Perform progressive tampering test for each model
    final_packages = None
    for model_index, model_id in enumerate(available_models, 1):
        print(f"\n====== [{model_index}/{model_count}] Progressive tampering test for model {model_id} ======")

        # Get total parameter count for the model
        total_params = len(all_models_data[model_id])
        print(f"Model {model_id} has {total_params} parameters in total")

        # Gradually increase number of tampered parameters, from 1 to all parameters
        for tamper_count in range(1, total_params + 1):
            print(f"\n--- Tampering first {tamper_count}/{total_params} parameters of model {model_id} ({tamper_count / total_params:.1%}) ---")

            try:
                # Reload tree and create server instance for each test to ensure starting from original state
                CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_10.tree")
                print(f"CHT load successfully {CHT}")
                cloud_server = ModelCloudServer(HE, CHT, all_models_data)
                client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)

                # # Client registers all model parameters
                # for mid, params in all_models_data.items():
                #     client.register_model(mid, params)

                # Measure time for getting tampered model
                start_time = time.time()
                # Always tamper parameters from 1 to current tamper_count
                tampered_package = cloud_server.get_model(model_id, tamper_param_size=tamper_count, honest=False)
                end_time = time.time()
                get_model_time = (end_time - start_time) * 1000

                # Measure verification time
                start_time = time.time()
                results = client.verify_model(tampered_package)
                end_time = time.time()
                verification_time = (end_time - start_time) * 1000

                # Calculate total time and storage cost
                total_time = get_model_time + verification_time
                storage_cost = get_size(tampered_package) / (1024 * 1024)

                # Calculate number of failed parameters
                failed_params = sum(1 for r in results['params'].values() if not r['valid'])

                # Verification result
                verification_result = 'Verification Successful' if results['overall']['valid'] else 'Verification Failed'

                # Output verification results
                print(f"Verification results:")
                print(f"  Signature verification: {'Success' if results['signature']['valid'] else 'Failed'}")
                print(f"  Model path verification: {'Success' if results['model_path']['valid'] else 'Failed'}")
                print(f"  Parameter verification: {len(results['params']) - failed_params} successful, {failed_params} failed")
                print(f"  Overall result: {verification_result}")
                print(f"  Get model time: {get_model_time:.2f} ms")
                print(f"  Verification time: {verification_time:.2f} ms")
                print(f"  Total time: {total_time:.2f} ms")
                print(f"  Storage cost: {storage_cost:.2f} MB")

                # Record data to CSV
                with open(f"../table/time_storage_costs_with_cloud_tamper_client_specify_model.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_id,
                        tamper_count,
                        total_time,
                        storage_cost
                    ])

                # Save last test package for return
                final_packages = tampered_package

            except Exception as e:
                print(f"Error: Exception occurred while testing model {model_id} with {tamper_count} tampered parameters: {str(e)}")
                print(f"Will skip this test point and continue")
                traceback.print_exc()
                continue

    print("\n====== All model progressive tampering tests completed ======")
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
