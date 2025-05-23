from initialization.setup import load_ecdsa_keys, load_HE_keys
from dual_verification_tree.CHT_utils import load_cht_keys
from simulator_client_cloud.model_verification_demo import ModelVerifier, ModelCloudServer
from dual_verification_tree.build_CHT import load_chameleon_hash_tree, save_chameleon_hash_tree
from level_homomorphic_encryption.encrypted_process_model import extract_data_from_hash_node
import csv
from level_homomorphic_encryption.encrypted_process_model import process_model_for_hash_tree
import time
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.util import generate_random_matrices
from level_homomorphic_encryption.encrypted_process_model import binary_mean_representation, extract_param_features
import pickle
import traceback

def get_mid_dic_items(dictionary, skip_start=1, skip_end=1):
    items_iter = iter(dictionary.items())
    for _ in range(skip_start):
        try:
            next(items_iter)
        except StopIteration:
            return {}

        take_count = len(dictionary) - skip_start - skip_end

        if take_count <= 0:
            return {}

        middle_items = list(itertools.islice(items_iter, take_count))

        return dict(middle_items)

def add_model():
    print("================add model================")

    # load signature keys, cht_keys_params, HE_keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)
    HE = load_HE_keys()

    # load CHT
    CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_10.tree")
    print(f"CHT load successfully {CHT}")

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

    # add model
    add_model_id_mapping = {}
    add_model_id = []
    add_models_data = {}
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/add_model_2_model_params.txt", 'r',
              encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            add_model_id.append(key)
            add_model_id_mapping[key] = value


    for model_id, pre_trained_model_path in add_model_id_mapping.items():
        add_models_data[model_id] = {}
        output_file, encrypted_model_param = process_model_for_hash_tree(pre_trained_model_path,
                                                                         "../level_homomorphic_encryption/model_hash_nodes")
        final_encrypted_model_param = get_mid_dic_items(encrypted_model_param)
        print(final_encrypted_model_param)
        for name, param in final_encrypted_model_param.items():
            add_models_data[model_id][name] = param

    # simulate the cloud server
    cloud_server = ModelCloudServer(HE, CHT, all_models_data)

    # create CSV file with header
    with open(f"../table/add_model_time_storage_costs.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_count', 'time_costs', 'storage_costs'])

    model_keys = list(add_model_id_mapping.keys())

    for i in range(1, len(model_keys) + 1, 1):
        start_time = time.time()
        print(f"Processing {i} models...")

        # Take first i items from all_models_data
        subset_models_data = {}
        for j in range(i):
            if j < len(model_keys):
                key = model_keys[j]
                subset_models_data[key] = add_models_data[key]

        update_tree = cloud_server.add_new_model(model_id_str=add_model_id[:i], model_params=subset_models_data)
        total_update_tree_size = update_tree.calculate_storage_size() / (1024 * 1024)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        save_chameleon_hash_tree(update_tree, f"../dual_verification_tree/tree/CHT_{10+i}.tree")

        with open(f"../table/add_model_time_storage_costs.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, total_time, total_update_tree_size])

    return update_tree

def delete_model():
    print("================delete model================")

    # load HE_keys
    HE = load_HE_keys()

    # load CHT
    CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_15.tree")
    print(f"CHT load successfully {CHT}")

    all_models_data = {}
    model_id_mapping = {}
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/update_model_id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        encrypted_model_param = extract_data_from_hash_node(encrypted_path)
        for name, param in encrypted_model_param.items():
            all_models_data[model_id][name] = param


    # simulate the cloud server
    cloud_server = ModelCloudServer(HE, CHT, all_models_data)

    # create CSV file with header
    with open(f"../table/delete_model_time_storage_costs.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['deleted_model_count', 'time_costs', 'storage_costs'])

    deleted_model_id = [["xcnn1"], ["xcnn2", "xcnn3"], ["xcnn4", "xcnn5", "cnn1"], ["cnn2", "cnn3", "cnn4", "cnn5"],
                        ["lenet1", "lenet5", "alexnet", "vgg16", "resnet18"]]

    for i in range(len(deleted_model_id)):
        start_time = time.time()
        update_tree = cloud_server.delete_model(deleted_model_id[i])
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        total_delete_tree_size = update_tree.calculate_storage_size() / (1024 * 1024)

        with open(f"../table/delete_model_time_storage_costs.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i+1, total_time, total_delete_tree_size])

    return update_tree


def modify_model_params():
    print("================modify model parameters================")

    # load HE_keys
    HE = load_HE_keys()

    # load model ID mapping
    all_models_data = {}
    model_id_mapping = {}
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/update_model_id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        encrypted_model_param = extract_data_from_hash_node(encrypted_path)
        for name, param in encrypted_model_param.items():
            all_models_data[model_id][name] = param

    modify_model_id_str = ['lenet1', 'lenet5', 'alexnet', 'vgg16', 'resnet18']
    modify_param_id_str = {}
    for model_id_str in modify_model_id_str:
        modify_param_id_str[model_id_str] = {}
        count = 0
        specify_model_params = all_models_data[model_id_str]
        for name, param in specify_model_params.items():
            if count >= 6:
                break
            modify_param_id_str[model_id_str][name] = param
            count += 1

    original_CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_15.tree")
    cloud_server = ModelCloudServer(HE, original_CHT, all_models_data)

    # build results csv file
    with open(f"../table/modify_model_params_time_storage_costs.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'modify_parameters_count', 'time_costs', 'storage_costs'])

        # perfom modify test for each model
    for model_id_str in modify_model_id_str:
        params = list(modify_param_id_str[model_id_str].keys())

        for modify_param_count in range(1, len(params) + 1):
            # prepare the modifing parameters
            final_modify_param = {}
            selected_params = params[0:modify_param_count]

            for param_id in selected_params:
                original_param = modify_param_id_str[model_id_str][param_id]
                # generate differnet random data based on parameters type
                if param_id.endswith("weight"):
                    modify_param = generate_random_matrices(count=64, fixed_shape=(3, 3), min_val=-1, max_val=1)
                else:
                    modify_param = generate_random_matrices(count=1, fixed_shape=(1, 64), min_val=-1, max_val=1)

                # deal with the parameters
                modify_param_binary_info = binary_mean_representation(modify_param)
                modify_binary_mean = modify_param_binary_info['binary_mean']
                modify_param_features = extract_param_features(modify_param, param_id)
                modify_param_features_bytes = pickle.dumps(modify_param_features)
                encrypted_modify_param_mean = HE.encode_number(modify_binary_mean)
                encrypted_modify_param_mean_bytes = encrypted_modify_param_mean.to_bytes()
                modify_param_feature_length = len(modify_param_features_bytes)

                # build the final modify parameters
                final_modify_param[param_id] = (
                        modify_param_feature_length.to_bytes(4, byteorder='big') +
                        modify_param_features_bytes +
                        encrypted_modify_param_mean_bytes
                )

            # scale the time and storage costs of modifying operation
            start_time = time.time()
            tree = cloud_server.modify_model_param(model_id_str, final_modify_param)
            end_time = time.time()

            if tree is not None:
                total_modify_model_params_size = tree.calculate_storage_size() / (1024 * 1024)
                total_time = (end_time - start_time) * 1000

                # record results
                with open(f"../table/modify_model_params_time_storage_costs.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_id_str, modify_param_count, total_time, total_modify_model_params_size])

                print(
                    f"modified model {model_id_str}'s {modify_param_count} parameters，the time costs is {total_time:.2f} ms，the storage costs is {total_modify_model_params_size:.2f} MB")
            else:
                print(f"modified model {model_id_str}'s {modify_param_count} parameters failure！")
    return tree


def modify_model_with_fixed_params():
    print("================modify model with fixed parameters================")

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

    
    available_models = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5']

   
    modify_param_id_str = {}
    for model_id_str in available_models:
        modify_param_id_str[model_id_str] = {}
        count = 0
        specify_model_params = all_models_data[model_id_str]
        for name, param in specify_model_params.items():
            if count >= 6:
                break
            modify_param_id_str[model_id_str][name] = param
            count += 1
        print(f"model {model_id_str} has prepared {count} parameters for mofification")


    csv_file_path = "../table/modify_multi_models_time_storage_costs.csv"
    with open(csv_file_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['models_count', 'time_costs', 'storage_costs'])

    # define experiment configuration
    experiment_configs = [
       
        {'name': 'test single model', 'models': ['cnn1']},

        
        {'name': 'test double models', 'models': ['cnn1', 'cnn2']},

        
        {'name': 'test triple models', 'models': ['cnn1', 'cnn2', 'cnn3']},

        
        {'name': 'test four models', 'models': ['cnn1', 'cnn2', 'cnn3', 'cnn4']},

        
        {'name': 'test five models', 'models': ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5']}
    ]

   
    final_tree = None
    for exp_id, experiment in enumerate(experiment_configs, 1):
        selected_models = experiment['models']
        models_name_str = '+'.join(selected_models)
        num_models = len(selected_models)

        print(f"\n===== experiment {exp_id}: {experiment['name']} - test modified {num_models} models: {models_name_str} =====")

       
        CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_10.tree")
        print(f"CHT load successfully {CHT}")
        cloud_server = ModelCloudServer(HE, CHT, all_models_data)

        
        all_modify_params = {}
        total_param_count = 0

       
        for model_id_str in selected_models:
           
            final_modify_param = {}
            model_params = list(modify_param_id_str[model_id_str].keys())

            
            for param_id in model_params:
                original_param = modify_param_id_str[model_id_str][param_id]

                
                if param_id.endswith("weight"):
                    modify_param = generate_random_matrices(count=64, fixed_shape=(3, 3), min_val=-1, max_val=1)
                else:
                    modify_param = generate_random_matrices(count=1, fixed_shape=(1, 64), min_val=-1, max_val=1)

                    
                modify_param_binary_info = binary_mean_representation(modify_param)
                modify_binary_mean = modify_param_binary_info['binary_mean']
                modify_param_features = extract_param_features(modify_param, param_id)
                modify_param_features_bytes = pickle.dumps(modify_param_features)
                encrypted_modify_param_mean = HE.encode_number(modify_binary_mean)
                encrypted_modify_param_mean_bytes = encrypted_modify_param_mean.to_bytes()
                modify_param_feature_length = len(modify_param_features_bytes)

                
                final_modify_param[param_id] = (
                        modify_param_feature_length.to_bytes(4, byteorder='big') +
                        modify_param_features_bytes +
                        encrypted_modify_param_mean_bytes
                )

                
            all_modify_params[model_id_str] = final_modify_param
            total_param_count += len(final_modify_param)
            print(f"prepare modifying {model_id_str}'s {len(final_modify_param)} parameters")

           
        start_time = time.time()


        tree = None
        try:
           
            for model_id, params in all_modify_params.items():
                print(f"modify{model_id}'s {len(params)} parameters...")
                tree = cloud_server.modify_model_param(model_id, params)
                if tree is None:
                    print(f"modified {model_id} failure！")
                    break
        except Exception as e:
            print(f"error of modifying model: {str(e)}")
            traceback.print_exc()

        end_time = time.time()

        if tree is not None:
            total_modify_models_size = tree.calculate_storage_size() / (1024 * 1024)
            total_time = (end_time - start_time) * 1000
            final_tree = tree  

            with open(csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    num_models,
                    total_time,
                    total_modify_models_size
                ])

              
            avg_time_per_param = total_time / total_param_count if total_param_count > 0 else 0

            print(f"modified {num_models} models {total_param_count} parameters，")
            print(f"total time costs is {total_time:.2f} ms，average {avg_time_per_param:.2f} ms for each parameter")
            print(f"total storage costs is {total_modify_models_size:.2f} MB")
        else:
            print(f"modifing {num_models} model parameters failure！")

    return final_tree

def draw_time_storage_costs_4cols(filepath):
    df = pd.read_csv(filepath)

    # Filter to include only the first 6 entries per model
    models = df['model_name'].unique()
    filtered_data = pd.DataFrame()

    for model in models:
        model_data = df[df['model_name'] == model].head(6)  # Get first 6 rows for each model
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
            model_rows = filtered_data[filtered_data['model_name'] == model]
            if i < len(model_rows):
                time_data.append(model_rows.iloc[i]['time_costs'])
            else:
                time_data.append(0)

        plt.bar(positions + (i - 2.5) * bar_width, time_data, bar_width,
                label=f'The model provider modified {i + 1} params', color=colors[i])

    plt.ylabel('Time costs of modifying model paramters in $\mathcal{DVT}$ (ms)',fontsize=12, fontweight='bold')
    plt.xlabel('Model name',fontsize=12, fontweight='bold')
    # plt.title('Time Costs by Model')
    plt.xticks(positions, models)
    plt.legend(title='The number of modified parameters of a model')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../figure/model_provider_modify_model_parameters_time_costs.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Second plot: Storage Costs by Model Name
    plt.figure(figsize=(10, 6))
    for i in range(6):  # For each of the 6 parameter counts
        storage_data = []
        for model in models:
            model_rows = filtered_data[filtered_data['model_name'] == model]
            if i < len(model_rows):
                storage_data.append(model_rows.iloc[i]['storage_costs'])
            else:
                storage_data.append(0)

        plt.bar(positions + (i - 2.5) * bar_width, storage_data, bar_width,
                label=f'The model provider modified {i + 1} params', color=colors[i])

    plt.ylabel('Storage costs of modifying model paramters in $\mathcal{DVT}$ (ms)',fontsize=12, fontweight='bold')
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    # plt.title('Storage Costs by Model')
    plt.xticks(positions, models)
    plt.legend(title='The number of modified parameters of a model')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../figure/model_provider_modify_model_parameters_storage_costs.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_time_storage_costs_3cols(filepath, csv_col1, csv_col2, csv_col3, save_file_path1, save_file_path2, xla, yla1, yla2):
    # Set style for better visualization
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Read the CSV file
    df = pd.read_csv(filepath)

    # Convert time from seconds to milliseconds
    df['time_costs_ms'] = df[csv_col2]

    # Figure 1: Time Costs vs Model Size
    plt.figure(figsize=(7, 5))
    time_bars = plt.bar(df[csv_col1], df['time_costs_ms'], color='mediumpurple', width=0.6)
    plt.xlabel(xla, fontsize=12, fontweight='bold')
    plt.ylabel(yla1, fontsize=12, fontweight='bold')
    # plt.title('Time Costs vs Model Size', fontsize=14, fontweight='bold')
    # plt.xticks(df['model_count'])
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of the bars
    # for bar in time_bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
    #              f'{height:.1f}',
    #              ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_file_path1, dpi=300, bbox_inches='tight')
    plt.show

    # Figure 2: Storage Costs vs Model Size
    plt.figure(figsize=(7, 5))
    storage_bars = plt.bar(df[csv_col1], df[csv_col3], color='hotpink', width=0.6)
    plt.xlabel(xla, fontsize=12, fontweight='bold')
    plt.ylabel(yla2, fontsize=12, fontweight='bold')
    # plt.title('Storage Costs vs Model Size', fontsize=14, fontweight='bold')
    # plt.xticks(df['model_count'])
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of the bars
    # for bar in storage_bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
    #              f'{height:.1f}',
    #              ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_file_path2, dpi=300, bbox_inches='tight')
    plt.show()  # Display the second figure (optional)

if __name__ == "__main__":
    # add_model()

    # delete_model()

    modify_model_with_fixed_params()



    # add_model_file_path = "../table/add_model_time_storage_costs.csv"
    # save_fig_path1 = "../figure/add_model_time_costs.png"
    # save_fig_path2 = "../figure/add_model_storage_costs.png"
    # draw_time_storage_costs_3cols(filepath=add_model_file_path, csv_col1="model_count",
    #                         csv_col2="time_costs", csv_col3="storage_costs",
    #                         save_file_path1=save_fig_path1, save_file_path2=save_fig_path2,
    #                         xla="Increased number of model", yla1="Time costs of adding models in $\mathcal {DVT}$ (ms)",
    #                         yla2="Storage costs of adding models in $\mathcal {DVT}$ (MB)")

    # delete_model_file_path = "../table/delete_model_time_storage_costs.csv"
    # save_fig_path1 = "../figure/delete_model_time_costs.png"
    # save_fig_path2 = "../figure/delete_model_storage_costs.png"
    # draw_time_storage_costs_3cols(filepath=delete_model_file_path, csv_col1="deleted_model_count",
    #                         csv_col2="time_costs", csv_col3="storage_costs",
    #                         save_file_path1=save_fig_path1, save_file_path2=save_fig_path2,
    #                         xla="Deleted number of model", yla1="Time costs of deleting models in $\mathcal {DVT}$ (ms)",
    #                         yla2="Storage costs of deleting models in $\mathcal {DVT}$ (MB)")

    # draw_time_storage_costs_4cols("../table/modify_model_params_time_storage_costs.csv")

    modify_model_file_path = "../table/modify_multi_models_time_storage_costs.csv"
    save_fig_path1 = "../figure/modify_multi_model_time_costs.png"
    save_fig_path2 = "../figure/modify_multi_model_storage_costs.png"
    draw_time_storage_costs_3cols(filepath=modify_model_file_path, csv_col1="models_count",
                            csv_col2="time_costs", csv_col3="storage_costs",
                            save_file_path1=save_fig_path1, save_file_path2=save_fig_path2,
                            xla="The number of models modified", yla1="Time costs of modifying models in $\mathcal {DVT}$ (ms)",
                            yla2="Storage costs of modifying models in $\mathcal {DVT}$ (MB)")
