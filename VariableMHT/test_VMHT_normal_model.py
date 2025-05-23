import numpy as np
from VariableMHT.build_VMHT import VariableMerkleHahsTree
import csv
import time
from utils.util import get_size
import traceback


def client_get_normal_model():
    all_models_data = {}
    model_id_mapping = {}
    # get model id
    with open("./model_id_pre_trained_model.txt", 'r',
              encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    print(model_id_mapping)
    # get encrypted model params
    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}

        # Load encrypted model parameters
        encrypted_data = np.load(encrypted_path, allow_pickle=True)

        # Handle different types of NumPy arrays
        if isinstance(encrypted_data, np.ndarray) and encrypted_data.dtype == np.dtype('O'):
            # Handle object arrays
            if encrypted_data.ndim == 0:
                # 0-dimensional object array - use item() to get the dictionary inside
                model_params = encrypted_data.item()
                if not isinstance(model_params, dict):
                    print(f"Warning: Data for model {model_id} is not in dictionary format")
                    model_params = {"parameters": model_params}
            else:
                # Multi-dimensional object array - usually the first element of the array
                if len(encrypted_data) > 0 and isinstance(encrypted_data[0], dict):
                    model_params = encrypted_data[0]
                else:
                    print(f"Warning: Data format for model {model_id} is not the expected dictionary array")
                    model_params = {"full_array": encrypted_data}
        else:
            # Not an object array, might be a direct numerical array
            print(f"Data for model {model_id} is in simple array format")
            model_params = {"parameters": encrypted_data}

        # Add parameters to all model data
        print(f"Processing model {model_id}, parameter count: {len(model_params)}")
        for name, param in model_params.items():
            all_models_data[model_id][name] = param
            if isinstance(param, np.ndarray):
                print(f"  Parameter {name}: shape {param.shape}, type {param.dtype}")

    vmht_builder = VariableMerkleHahsTree()
    # Increasing target_chunks can increase tree building overhead
    VMHT, performance = vmht_builder.build_tree(all_models_data)

    with open(f"./client_verification_time_storage_costs_untamper.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'time_costs', 'storage_costs', 'verification_success', 'total_blocks'])

    for model_id_str in model_id_mapping.keys():
        start_time = time.time()
        client_request_model = model_id_str
        print(f"the client request {client_request_model} model")

        # Using parameter block verification method
        try:
            # Generate verification request
            request = vmht_builder.generate_param_verification_request(client_request_model)

            # Generate parameter proof
            proof = vmht_builder.generate_param_proof(client_request_model)

            model_package_size = get_size(proof) / (1024 * 1024)

            print("\nthe client starts run verification operation:")
            # Verify all parameter blocks
            results = VMHT.verify_params(proof, request)

            end_time = time.time()
            total_time = (end_time - start_time) * 1000

            with open(f"./client_verification_time_storage_costs_untamper.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_id_str,
                    total_time,
                    model_package_size,
                    "Success" if results["success"] else "Fail",
                    results["total_count"]
                ])

            print(f"record experiment data: model name={model_id_str}, "
                  f"total time={total_time}ms, "
                  f"model verification package size={model_package_size}MB, "
                  f"success rate={results['success_rate'] * 100:.2f}%, "
                  f"verified {results['success_count']}/{results['total_count']} blocks")

        except Exception as e:
            print(f"Error verifying model {model_id_str}: {str(e)}")
            traceback.print_exc()

    return all_models_data


if __name__ == "__main__":
    model_datas = client_get_normal_model()
