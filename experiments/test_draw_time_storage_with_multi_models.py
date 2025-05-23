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
    Read data from CSV file and draw bar charts for time and storage costs of models under different schemes

    Parameters:
    file_path -- CSV file path
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return

    # Read data from file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    # Check if required columns exist
    required_columns = ['schemes', 'model_name', 'time_costs', 'storage_costs']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: CSV file missing required column '{col}'")
            return

    # Convert numeric columns to float
    df['time_costs'] = pd.to_numeric(df['time_costs'], errors='coerce')
    df['storage_costs'] = pd.to_numeric(df['storage_costs'], errors='coerce')

    # Set font
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Get unique schemes and model names
    schemes = sorted(df['schemes'].unique())
    model_names = df['model_name'].unique()

    # Set different colors for each scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors
    width = 0.25  # Bar width

    # Create x-axis positions
    x = np.arange(len(model_names))

    # ====== Time Cost Chart ======
    plt.figure(figsize=(7, 5))

    # Draw time bar chart
    for i, scheme in enumerate(schemes):
        scheme_data = df[df['schemes'] == scheme]

        # Ensure data is sorted by model name
        scheme_data = pd.merge(
            pd.DataFrame({'model_name': model_names}),
            scheme_data,
            on='model_name',
            how='left'
        ).fillna(0)

        plt.bar(x + (i - len(schemes) / 2 + 0.5) * width, scheme_data['time_costs'],
                width, label=scheme, color=colors[i % len(colors)])

    # Add labels and title
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    plt.ylabel('Verification time costs (ms)', fontsize=12, fontweight='bold')
    # plt.title('Time Costs for Different Schemes and Models', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, fontsize=10)
    plt.yticks(fontsize=10)

    # Create log scale to better display small values
    plt.yscale('log')

    # Format y-axis ticks for better readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    # Add legend
    plt.legend(fontsize=10)

    # Use tight layout
    plt.tight_layout()

    # Create output directory (if it doesn't exist)
    output_dir = '../figure'
    os.makedirs(output_dir, exist_ok=True)

    # Save image
    time_output_path = os.path.join(output_dir, '../figure/verification_time_costs_with_different_models_and_schemes.png')
    plt.savefig(time_output_path, dpi=300, bbox_inches='tight')
    print(f"Time cost chart saved to: {time_output_path}")

    # Display image
    plt.show()

    # ====== Storage Cost Chart ======
    plt.figure(figsize=(7, 5))

    # Draw storage bar chart
    for i, scheme in enumerate(schemes):
        scheme_data = df[df['schemes'] == scheme]

        # Ensure data is sorted by model name
        scheme_data = pd.merge(
            pd.DataFrame({'model_name': model_names}),
            scheme_data,
            on='model_name',
            how='left'
        ).fillna(0)

        plt.bar(x + (i - len(schemes) / 2 + 0.5) * width, scheme_data['storage_costs'],
                width, label=scheme, color=colors[i % len(colors)])

    # Add labels and title
    plt.xlabel('Model name', fontsize=12, fontweight='bold')
    plt.ylabel('Verification storage costs (MB)', fontsize=12, fontweight='bold')
    # plt.title('Storage Costs for Different Schemes and Models', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, fontsize=10)
    plt.yticks(fontsize=10)

    # Create log scale to better display small values
    plt.yscale('log')

    # Format y-axis ticks for better readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.1f}'.format(x)))

    # Add legend
    plt.legend(fontsize=10)

    # Use tight layout
    plt.tight_layout()

    # Save image
    storage_output_path = os.path.join(output_dir, '../figure/verification_storage_costs_with_different_models_and_schemes.png')
    plt.savefig(storage_output_path, dpi=300, bbox_inches='tight')
    print(f"Storage cost chart saved to: {storage_output_path}")

    # Display image
    plt.show()


# Usage example
if __name__ == "__main__":
    # Specify CSV file path
    file_path = '../table/verification_time_storage_costs_comparison.txt'
    draw_verification_costs_with_different_schemes(file_path)
