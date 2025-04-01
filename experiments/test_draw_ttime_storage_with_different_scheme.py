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


def draw_build_data_structure_time_costs_with_different_scheme(file_path):
    df = pd.read_csv(file_path)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Get unique values
    client_counts = sorted(df['model_count'].unique())
    x = np.arange(len(client_counts))
    schemes = df['scheme'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors
    width = 0.25  # Bar width

    # TIME COSTS CHART
    plt.figure(figsize=(10, 6))

    # Draw time bar chart
    for i, scheme in enumerate(schemes):
        scheme_data = df[df['scheme'] == scheme]
        # Ensure data is sorted by client_count
        scheme_data = scheme_data.sort_values(by='model_count')
        plt.bar(x + (i - 1) * width, scheme_data['time_costs'], width, label=scheme, color=colors[i])

        # Add labels and title
    plt.xlabel('Number of models', fontsize=12, fontweight='bold')
    plt.ylabel('The time costs of building verification tree (ms)', fontsize=12, fontweight='bold')
    plt.xticks(x, client_counts, fontsize=12)
    plt.yticks(fontsize=12)

    # Create log scale for y-axis to better show small values
    plt.yscale('log')

    # Format y-axis ticks for better readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    # Add legend
    plt.legend(fontsize=12)

    # Use tight layout
    plt.tight_layout()

    # Save image
    plt.savefig('../figure/build_data_structure_time_costs_with_different_schemes.png', dpi=300, bbox_inches='tight')

    # Show image
    plt.show()

    # STORAGE COSTS CHART
    plt.figure(figsize=(10, 6))

    # Draw storage bar chart
    for i, scheme in enumerate(schemes):
        scheme_data = df[df['scheme'] == scheme]
        # Ensure data is sorted by client_count
        scheme_data = scheme_data.sort_values(by='model_count')
        plt.bar(x + (i - 1) * width, scheme_data['storage_costs'], width, label=scheme, color=colors[i])

        # Add labels and title
    plt.xlabel('Number of models', fontsize=12, fontweight='bold')
    plt.ylabel('The storage costs of building verification tree (MB)', fontsize=12, fontweight='bold')
    plt.xticks(x, client_counts, fontsize=12)
    plt.yticks(fontsize=12)

    # Create log scale for y-axis to better show small values
    plt.yscale('log')

    # Format y-axis ticks for better readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.1f}'.format(x)))

    # Add legend
    plt.legend(fontsize=12)

    # Use tight layout
    plt.tight_layout()

    # Save image
    plt.savefig('../figure/build_data_structure_storage_costs_with_different_schemes.png', dpi=300, bbox_inches='tight')

    # Show image
    plt.show()


if __name__ == "__main__":
    draw_build_data_structure_time_costs_with_different_scheme("../table/build_tree_comparison.txt")