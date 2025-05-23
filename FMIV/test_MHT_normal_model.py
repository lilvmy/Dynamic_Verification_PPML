
import numpy as np
from FMIV.build_MHT import load_merkle_hash_tree, MerkleHashTree
import csv
import time
from utils.util import get_size


def client_get_normal_model():
    all_models_data = {}
    model_id_mapping = {}
    # get model id
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/FMIV/model_id_pre_trained_model.txt", 'r',
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

    mht_builder = MerkleHashTree()
    # Increasing target_chunks can increase tree building overhead
    MHT, performance = mht_builder.build_from_model_params(all_models_data, target_chunks=18024)



    with open(f"./client_verification_time_storage_costs_untamper.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'time_costs', 'storage_costs'])

    for model_id_str in model_id_mapping.keys():
        start_time = time.time()
        client_request_model = model_id_str
        print(f"the client request {client_request_model} model")

        model_package = MHT.get_model_proof(client_request_model)
        model_package_size = get_size(model_package) / (1024 * 1024)

        print("\nthe client starts run verification operation:")
        results = MHT.verify_model_proof(model_package)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        with open(f"./client_verification_time_storage_costs_untamper.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_id_str, total_time, model_package_size])

        print(f"record experiment data: model name={model_id_str}, total time={total_time}ms, model verification package size={model_package_size}MB")


    return all_models_data

if __name__ == "__main__":
    model_datas = client_get_normal_model()
