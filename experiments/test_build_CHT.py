from initialization.setup import load_ecdsa_keys
from dual_verification_tree.CHT_utils import load_cht_keys
from dual_verification_tree.build_CHT import save_chameleon_hash_tree
from level_homomorphic_encryption.encrypted_process_model import process_model_for_hash_tree, extract_data_from_hash_node
from dual_verification_tree.build_CHT import ChameleonHashTree
import time
import csv
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def draw_time_storage_costs(filepath):
    # set style for better visualization
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Read CSV file
    df = pd.read_csv(filepath)
    print("CSV columns:", df.columns.tolist())

    # Define color mapping for all possible models
    model_colors = {
        'cnn1': '#1f77b4',  # blue
        'cnn2': '#ff7f0e',  # orange
        'cnn3': '#2ca02c',  # green
        'cnn4': '#d62728',  # red
        'cnn5': '#9467bd',  # purple
        'lenet1': '#8c564b',  # brown
        'lenet5': '#e377c2',  # pink
        'alexnet': '#7f7f7f',  # gray
        'vgg16': '#bcbd22',  # olive green
        'resnet18': '#17becf'  # cyan
    }

    # figure 1: time costs vs model name with stacked bars
    plt.figure(figsize=(7, 5))

    # Get x-axis positions
    x_pos = np.arange(len(df))
    width = 0.2

    for idx, row in df.iterrows():
        # Split model names into list
        models_in_bar = row['model_name'].split('+')
        # Calculate share for each model (equally divide time costs)
        model_share = row['time_costs'] / len(models_in_bar) * 1000
        # Initialize bottom position
        bottom = 0

        # Draw stacked bar segment for each model
        for model in models_in_bar:
            plt.bar(idx, model_share, width, bottom=bottom,
                    color=model_colors.get(model, 'gray'),
                    label=model if model not in plt.gca().get_legend_handles_labels()[1] else "")
            bottom += model_share

    plt.xlabel('The number of models', fontsize=12, fontweight='bold')
    plt.ylabel('Time costs of building $\mathcal {DVT}$ (ms)', fontsize=12, fontweight='bold')

    # Use model_size as x-axis labels
    plt.xticks(x_pos, [f'{size}' for size in df['model_size']])

    # Create legend, but only show unique model names
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Model name",
               loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig('../figure/time_costs_of_building_CHT.png', dpi=300, bbox_inches='tight')
    plt.show()

    # figure 2: storage costs vs model size with stacked bars
    plt.figure(figsize=(7, 5))

    for idx, row in df.iterrows():
        # Split model names into list
        models_in_bar = row['model_name'].split('+')
        # Calculate share for each model (equally divide storage costs)
        model_share = row['storage_costs'] / len(models_in_bar)
        # Initialize bottom position
        bottom = 0

        # Draw stacked bar segment for each model
        for model in models_in_bar:
            plt.bar(idx, model_share, width, bottom=bottom,
                    color=model_colors.get(model, 'gray'),
                    label=model if model not in plt.gca().get_legend_handles_labels()[1] else "")
            bottom += model_share

    plt.xlabel('The number of models', fontsize=12, fontweight='bold')
    plt.ylabel('Storage costs of building $\mathcal{DVT}$ (MB)', fontsize=12, fontweight='bold')

    # Use model_size as x-axis labels
    plt.xticks(x_pos, [f'{size}' for size in df['model_size']])

    # Create legend, but only show unique model names
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Model name",
               loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig('../figure/storage_costs_of_building_CHT.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # load cht_keys_params
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)

    # load ecdsa keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()

    all_models_data = {}
    model_id_mapping = {}
    # get model id
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/model_id_2_pre_trained_model.txt", 'r',
              encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value
    print(model_id_mapping)

    # get encrypted model params
    for model_id, pre_trained_model_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        output_file, encrypted_model_param = process_model_for_hash_tree(pre_trained_model_path,
                                                                         "../level_homomorphic_encryption/model_hash_nodes")
        final_encrypted_model_param = get_mid_dic_items(encrypted_model_param)
        print(final_encrypted_model_param)
        for name, param in final_encrypted_model_param.items():
            all_models_data[model_id][name] = param

    # Create CSV file with header
    with open(f"../table/CHT_time_storage_costs.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_size', 'time_costs', 'storage_costs'])

    # Process models in increments of 2
    model_keys = list(all_models_data.keys())

    for i in range(2, len(model_keys) + 1, 2):
        print(f"Processing {i} models...")

        # Take first i items from all_models_data
        subset_models_data = {}
        for j in range(i):
            if j < len(model_keys):
                key = model_keys[j]
                subset_models_data[key] = all_models_data[key]

        # Build model verification tree CHT for this subset
        start_time = time.time()
        model_tree = ChameleonHashTree(cht_keys)
        model_tree.build_from_model_params(subset_models_data, ecdsa_private_key)
        total_tree_size = model_tree.calculate_storage_size() / (1024 * 1024)
        end_time = time.time()
        total_time = end_time - start_time

        # Save the tree
        save_chameleon_hash_tree(model_tree, f"../dual_verification_tree/tree/CHT_{i}.tree")

        # Append to CSV
        with open(f"../table/CHT_time_storage_costs.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, total_time, total_tree_size])

        print(f"Recorded experiment data: model count={i}, runtime={total_time:.4f}s, storage size={total_tree_size:.2f}MB")

    return model_tree

if __name__ == "__main__":
    # main()

    # draw time and storage costs for building CHT
    draw_time_storage_costs("../table/CHT_time_storage_costs.csv")
