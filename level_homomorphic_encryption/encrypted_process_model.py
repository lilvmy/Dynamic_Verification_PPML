import base64
import pickle
import numpy as np
import os
import gc
import time
import psutil
import tempfile
import sys
import zlib
import shutil
import hashlib
from initialization.setup import load_HE_keys


def get_memory_usage():
    """
    Return memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def binary_mean_representation(param):
    """
    Represent parameter in binary form and calculate mean
    """
    # Binary representation (>0 is 1, otherwise 0)
    binary = (param > 0).astype(np.float32)

    # Calculate mean
    binary_mean = np.mean(binary)

    # Calculate mean of positive and negative values for parameter reconstruction
    if np.any(param > 0):
        positive_mean = np.mean(param[param > 0])
    else:
        positive_mean = 0.1  # Default value

    if np.any(param <= 0):
        negative_mean = np.mean(param[param <= 0])
    else:
        negative_mean = -0.1  # Default value

    return {
        'binary_mean': binary_mean,
        'positive_mean': positive_mean,
        'negative_mean': negative_mean,
        'shape': param.shape
    }


def extract_param_features(param, name):
    """
    Extract parameter features to ensure different parameters have unique identifiers
    """
    # Basic statistical information
    features = {
        'name': name,
        'shape': param.shape,
        'size': param.size,
        'min': float(np.min(param)),
        'max': float(np.max(param)),
        'mean': float(np.mean(param)),
        'std': float(np.std(param)),
        'median': float(np.median(param)),
        'sparsity': float(np.sum(param == 0) / param.size)
    }

    # Add percentile information
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        features[f'percentile_{p}'] = float(np.percentile(param, p))

    # Add histogram features
    hist, bin_edges = np.histogram(param, bins=10)
    features['histogram'] = hist.tolist()
    features['histogram_bins'] = bin_edges.tolist()

    # Add hash of first few and last few elements (for uniqueness)
    first_elements = param.flatten()[:20].tolist()
    last_elements = param.flatten()[-20:].tolist()
    features['sample_elements'] = first_elements + last_elements

    return features


def process_model_for_hash_tree(model_path, output_dir, compression_level=9):
    """
    Process model parameters for chameleon hash tree leaf nodes, ensuring each model parameter has unique identifier
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Loading HE keys...")
    HE = load_HE_keys()

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = np.load(f, allow_pickle=True).item()

    param_names = list(model.keys())
    print(f"Found {len(param_names)} parameters")

    # Build processed model parameter dictionary
    processed_model = {}

    # Create model fingerprint for global identification
    model_fingerprint = hashlib.sha256()
    for name, param in model.items():
        # Add parameter's key statistical information to model fingerprint
        param_info = f"{name}:{param.shape}:{np.mean(param)}:{np.std(param)}"
        model_fingerprint.update(param_info.encode())

    # Global unique identifier for the model
    model_id = model_fingerprint.hexdigest()
    processed_model['__model_id__'] = model_id
    print(f"Model ID: {model_id}")

    # Process each parameter
    for name in param_names:
        print(f"Processing parameter: {name}")
        param = model[name]

        # Get binary mean information
        binary_info = binary_mean_representation(param)
        binary_mean = binary_info['binary_mean']

        print(f"  Shape: {binary_info['shape']}, Elements: {param.size}")
        print(f"  Binary mean: {binary_mean:.4f}")
        print(f"  Positive elements mean: {binary_info['positive_mean']:.4f}")
        print(f"  Negative elements mean: {binary_info['negative_mean']:.4f}")

        # Extract parameter features
        param_features = extract_param_features(param, name)

        # Convert features to bytes
        feature_bytes = pickle.dumps(param_features)

        # Represent each parameter as binary mean and encrypt
        encrypted_mean = HE.encode_number(binary_mean)
        encrypted_bytes = encrypted_mean.to_bytes()

        # Calculate feature bytes length
        feature_length = len(feature_bytes)
        print(f"  Feature data size: {feature_length} bytes")
        print(f"  Encrypted data size: {len(encrypted_bytes)} bytes")

        # Combine features and encrypted data, first 4 bytes store feature length
        combined_data = feature_length.to_bytes(4, byteorder='big') + feature_bytes + encrypted_bytes

        # Store combined data
        processed_model[name] = combined_data

    # Add model structure information
    model_structure = {name: str(param.shape) for name, param in model.items()}
    processed_model['__model_structure__'] = pickle.dumps(model_structure)

    model_name = os.path.basename(model_path).replace('.npy', '')
    output_file = os.path.join(output_dir, f"{model_name}_hash_node.zpkl")

    # Use temporary file to save, then move to output file path
    temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp')
    try:
        with os.fdopen(temp_fd, 'wb') as temp_f:
            # Compress data
            compressed_data = zlib.compress(pickle.dumps(processed_model), level=compression_level)
            temp_f.write(compressed_data)

        # Move file
        shutil.move(temp_path, output_file)
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

    # Calculate file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Processed model saved for hash tree. File size: {file_size_mb:.2f} MB")

    return output_file, processed_model


def cleanup_temp_files():
    """
    Clean up all temporary files
    """
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        if filepath.endswith('.tmp'):
            try:
                os.unlink(filepath)
            except Exception:
                pass
    gc.collect()


def verify_hash_node_files(output_dir):
    """
    Verify that generated hash node files have uniqueness
    """
    print("\n=== Verifying Hash Node Files ===")
    file_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_hash_node.zpkl')]

    model_ids = {}
    param_hashes = {}

    for file_path in file_paths:
        model_name = os.path.basename(file_path).replace('_hash_node.zpkl', '')

        try:
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
                model_data = pickle.loads(zlib.decompress(compressed_data))

            # Check model ID uniqueness
            model_id = model_data.get('__model_id__', 'unknown')
            if model_id in model_ids:
                print(f"WARNING: Model ID collision between {model_name} and {model_ids[model_id]}")
            else:
                model_ids[model_id] = model_name

            # Check parameter uniqueness
            for param_name, param_data in model_data.items():
                if param_name.startswith('__'):  # Skip metadata
                    continue

                # Extract parameter features
                try:
                    feature_length = int.from_bytes(param_data[:4], byteorder='big')
                    feature_bytes = param_data[4:4 + feature_length]
                    param_features = pickle.loads(feature_bytes)

                    # Create parameter feature hash
                    param_hash = hashlib.md5(feature_bytes).hexdigest()
                    param_key = f"{param_name}:{param_hash[:8]}"

                    if param_key in param_hashes:
                        print(f"  Parameter similarity: {model_name}.{param_name} ~ {param_hashes[param_key]}")
                    else:
                        param_hashes[param_key] = f"{model_name}.{param_name}"

                except Exception as e:
                    print(f"  Error processing {model_name}.{param_name}: {e}")

            print(f"Verified {model_name}: Model ID {model_id[:8]}..., {len(model_data) - 2} parameters")

        except Exception as e:
            print(f"Error verifying {model_name}: {e}")

    print(f"Found {len(model_ids)} unique model IDs across {len(file_paths)} files")


def main():
    models = [
        "lenet1_model_params.npy",
        "lenet5_model_params.npy",
        "cnn1_model_params.npy",
        "cnn2_model_params.npy",
        "cnn3_model_params.npy",
        "cnn4_model_params.npy",
        "cnn5_model_params.npy"
    ]

    # Create output file paths
    output_dir = "./model_hash_nodes"
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for model_name in models:
        model_path = f"../pre-trained-model/model/{model_name}"
        print(f"\nProcessing model: {model_name}")

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            results[model_name] = {'status': 'not_found'}
            continue

        try:
            output_file, combined_data = process_model_for_hash_tree(
                model_path,
                output_dir,
                compression_level=9
            )

            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            results[model_name] = {
                'status': 'success',
                'output_file': output_file,
                'file_size_mb': file_size_mb
            }

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }

        cleanup_temp_files()
        gc.collect()

    # Print results summary
    print("\n=== Processing Summary ===")
    for model, result in results.items():
        status = result['status']
        if status == 'success':
            print(f"{model}: Successfully processed for hash tree")
            print(f"  Output file: {result['output_file']}")
            print(f"  File size: {result['file_size_mb']:.2f} MB")
        elif status == 'failed':
            print(f"{model}: Failed - {result.get('error', 'Unknown error')}")
        else:
            print(f"{model}: Not found")

    # Verify generated hash node files
    verify_hash_node_files(output_dir)

    print("\nAll models have been processed!")


def extract_data_from_hash_node(file_path, param_name=None, include_identifier=True):
    """
    Extract parameter encrypted data from hash node file

    Parameters:
        file_path: Hash node file path
        param_name: Specified parameter name, if None returns encrypted data for all parameters
        include_identifier: Whether to include unique identifier in encrypted data (default True)

    Returns:
        If param_name specified: Returns features and encrypted data for single parameter
        If param_name is None: Returns encrypted data dictionary for all parameters, each encrypted data contains unique identifier
    """
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
        model_data = pickle.loads(zlib.decompress(compressed_data))

    # If parameter name is specified, only return data for that parameter
    if param_name is not None:
        if param_name not in model_data:
            print(f"Parameter {param_name} not found in {file_path}")
            return None

        param_data = model_data[param_name]

        # Extract features and encrypted data
        feature_length = int.from_bytes(param_data[:4], byteorder='big')
        feature_bytes = param_data[4:4 + feature_length]
        encrypted_bytes = param_data[4 + feature_length:]

        features = pickle.loads(feature_bytes)

        # If need to include identifier (for chameleon hash), add partial feature hash to encrypted data
        if include_identifier:
            # Create unique identifier
            param_id = hashlib.sha256(feature_bytes).digest()[:16]  # Use 16 bytes identifier
            encrypted_bytes = param_id + encrypted_bytes

        return {
            'features': features,
            'encrypted_bytes': encrypted_bytes
        }

    # If no parameter name specified, return encrypted data for all parameters
    else:
        result = {}
        # Get model ID as additional identifier
        model_id = model_data.get('__model_id__', b'').encode()[:8]

        for key, param_data in model_data.items():
            # Skip metadata fields
            if key.startswith('__'):
                continue

            try:
                # Extract encrypted data and features
                feature_length = int.from_bytes(param_data[:4], byteorder='big')
                feature_bytes = param_data[4:4 + feature_length]
                encrypted_bytes = param_data[4 + feature_length:]

                # If need to include unique identifier, add parameter feature hash to encrypted data
                if include_identifier:
                    # Create unique identifier for parameter (using hash of parameter name and features)
                    key_bytes = key.encode()
                    param_hash = hashlib.sha256(key_bytes + feature_bytes).digest()[:16]

                    # # Combine model ID, parameter ID and encrypted data
                    # unique_bytes = model_id + param_hash + encrypted_bytes
                    unique_bytes = param_hash
                    result[key] = unique_bytes
                else:
                    # Only return encrypted data
                    result[key] = encrypted_bytes

            except Exception as e:
                print(f"Error extracting data for parameter {key}: {e}")
                result[key] = None

        return result

if __name__ == "__main__":
    # try:
    #     main()
    # finally:
    #     cleanup_temp_files()

    all_encrypted_data = extract_data_from_hash_node("./model_hash_nodes/lenet1_model_params_hash_node.zpkl")
    print(all_encrypted_data)
