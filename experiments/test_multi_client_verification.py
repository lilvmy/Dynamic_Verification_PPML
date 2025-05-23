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

    # Specify model ID to verify
    specify_model_id = "lenet1"

    # Get model verification package - only get once so all clients use the same verification package
    model_package = cloud_server.get_model(specify_model_id)

    # Create CSV file and write header
    with open(f"../table/multi_client_verification_time_costs_threaded.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['client_count', 'avg_time'])

    # Test different numbers of clients
    client_counts = [20, 40, 60, 80, 100]
    results = {}

    # Calculate maximum concurrent threads to avoid creating too many threads
    max_threads = min(100, max(client_counts))

    # Define a thread-safe list to store verification times
    verification_times_lock = threading.Lock()

    # Define client verification function
    def client_verify(client, model_pkg, verification_times):
        start_time = time.perf_counter()
        verification_result = client.verify_model(model_pkg)
        end_time = time.perf_counter()

        verification_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Thread-safely add verification time
        with verification_times_lock:
            verification_times.append(verification_time)

        return verification_time

    for num_clients in client_counts:
        # Create specified number of clients
        clients = []
        for i in range(num_clients):
            client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)
            clients.append(client)

        # Calculate concurrency factor - adjust concurrency based on client count
        concurrency_factor = min(num_clients, max_threads)

        verification_times = []
        total_start_time = time.perf_counter()
        completed_count = 0

        # Use thread pool to execute client verification
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_factor) as executor:
            # Submit all client verification tasks
            future_to_client = {
                executor.submit(client_verify, client, model_package, verification_times): i
                for i, client in enumerate(clients)
            }

            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_client):
                client_idx = future_to_client[future]
                try:
                    verification_time = future.result()
                    completed_count += 1
                    # Print progress every 25% of completed clients
                    if completed_count % max(1, num_clients // 4) == 0:
                        print(f"Completed {completed_count}/{num_clients} client verifications")
                except Exception as e:
                    print(f"Error occurred during client {client_idx} verification: {e}")
                    completed_count += 1

        total_end_time = time.perf_counter()
        total_time = (total_end_time - total_start_time) * 1000  # Convert to milliseconds

        # Calculate statistics
        avg_time = sum(verification_times) / len(verification_times)
        min_time = min(verification_times)
        max_time = max(verification_times)

        # Save results
        results[num_clients] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_time': total_time,
            'concurrency_factor': concurrency_factor
        }

        # Create CSV file and write header
        with open(f"../table/multi_client_verification_time_costs_threaded.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([num_clients, min_time])

    # Print detailed results
    print("\n==== Detailed Statistical Results ====")
    print("Client Count\tAvg Time(ms)\tMin Time(ms)\tMax Time(ms)\tTotal Time(ms)\tConcurrency")
    for num_clients, stats in results.items():
        print(
            f"{num_clients}\t\t{stats['avg_time']:.2f}\t\t{stats['min_time']:.2f}\t\t{stats['max_time']:.2f}\t\t{stats['total_time']:.2f}\t\t{stats['concurrency_factor']}")

    return results


def draw_time_vs_client_count_from_file(file_path):
    """
    Read data from file and draw bar chart showing time vs client count

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
    required_columns = ['client_count', 'avg_time']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: CSV file missing required column '{col}'")
            return

    # Set font style
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Create figure
    plt.figure(figsize=(10, 6))

    # Get client count and time data
    client_counts = df['client_count'].tolist()
    avg_times = df['avg_time'].tolist()

    # Set x-axis positions
    x = np.arange(len(client_counts))
    width = 0.6  # Bar width

    # Draw bar chart
    bars = plt.bar(x, avg_times, width, color='#1f77b4')

    # # Add value labels on bars
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
    #              f'{height:.2f}',
    #              ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.xlabel('Number of clients', fontsize=12, fontweight='bold')
    plt.ylabel('Average verification time cost (ms)', fontsize=12, fontweight='bold')
    # plt.title('Average Time vs. Number of Clients', fontsize=14, fontweight='bold')

    # Set x-axis labels
    plt.xticks(x, client_counts, fontsize=12)
    plt.yticks(fontsize=12)

    # # Add grid lines to make chart more readable
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Create output directory (if it doesn't exist)
    output_dir = '../figure'
    os.makedirs(output_dir, exist_ok=True)

    # Save image
    output_path = os.path.join(output_dir, '../figure/avg_time_vs_client_count.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")

    # Display chart
    plt.show()

if __name__ == "__main__":
    # res = get_verification_time_costs_with_multi_clients()
    # print(res)

    draw_time_vs_client_count_from_file("../table/multi_client_verification_time_costs_threaded.csv")
