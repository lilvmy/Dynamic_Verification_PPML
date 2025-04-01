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
    返回内存使用量(MB)
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def binary_mean_representation(param):
    """
    将参数表示为二进制形式并计算均值
    """
    # 二进制表示(>0为1，否则为0)
    binary = (param > 0).astype(np.float32)

    # 计算均值
    binary_mean = np.mean(binary)

    # 计算正值和负值的均值用于重建参数
    if np.any(param > 0):
        positive_mean = np.mean(param[param > 0])
    else:
        positive_mean = 0.1  # 默认值

    if np.any(param <= 0):
        negative_mean = np.mean(param[param <= 0])
    else:
        negative_mean = -0.1  # 默认值

    return {
        'binary_mean': binary_mean,
        'positive_mean': positive_mean,
        'negative_mean': negative_mean,
        'shape': param.shape
    }


def extract_param_features(param, name):
    """
    提取参数的特征，确保不同参数有唯一标识
    """
    # 基本统计信息
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

    # 添加分位数信息
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        features[f'percentile_{p}'] = float(np.percentile(param, p))

        # 添加直方图特征
    hist, bin_edges = np.histogram(param, bins=10)
    features['histogram'] = hist.tolist()
    features['histogram_bins'] = bin_edges.tolist()

    # 添加前几个和最后几个元素的哈希（用于唯一性）
    first_elements = param.flatten()[:20].tolist()
    last_elements = param.flatten()[-20:].tolist()
    features['sample_elements'] = first_elements + last_elements

    return features


def process_model_for_hash_tree(model_path, output_dir, compression_level=9):
    """
    处理模型参数用于chameleon hash tree的叶子节点，确保每个模型参数有唯一标识
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print("Loading HE keys...")
    HE = load_HE_keys()

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = np.load(f, allow_pickle=True).item()

    param_names = list(model.keys())
    print(f"Found {len(param_names)} parameters")

    # 构建处理后的模型参数字典
    processed_model = {}

    # 创建模型指纹用于全局标识
    model_fingerprint = hashlib.sha256()
    for name, param in model.items():
        # 将参数的关键统计信息添加到模型指纹
        param_info = f"{name}:{param.shape}:{np.mean(param)}:{np.std(param)}"
        model_fingerprint.update(param_info.encode())

    # 模型的全局唯一标识
    model_id = model_fingerprint.hexdigest()
    processed_model['__model_id__'] = model_id
    print(f"Model ID: {model_id}")

    # 处理每个参数
    for name in param_names:
        print(f"Processing parameter: {name}")
        param = model[name]

        # 获取二进制均值信息
        binary_info = binary_mean_representation(param)
        binary_mean = binary_info['binary_mean']

        print(f"  Shape: {binary_info['shape']}, Elements: {param.size}")
        print(f"  Binary mean: {binary_mean:.4f}")
        print(f"  Positive elements mean: {binary_info['positive_mean']:.4f}")
        print(f"  Negative elements mean: {binary_info['negative_mean']:.4f}")

        # 提取参数特征
        param_features = extract_param_features(param, name)

        # 将特征转换为字节
        feature_bytes = pickle.dumps(param_features)

        # 将每个参数表示为二进制均值，并进行加密
        encrypted_mean = HE.encode_number(binary_mean)
        encrypted_bytes = encrypted_mean.to_bytes()

        # 计算特征字节长度
        feature_length = len(feature_bytes)
        print(f"  Feature data size: {feature_length} bytes")
        print(f"  Encrypted data size: {len(encrypted_bytes)} bytes")

        # 将特征和加密数据合并，前4字节存储特征长度
        combined_data = feature_length.to_bytes(4, byteorder='big') + feature_bytes + encrypted_bytes

        # 存储合并后的数据
        processed_model[name] = combined_data

    # 添加模型结构信息
    model_structure = {name: str(param.shape) for name, param in model.items()}
    processed_model['__model_structure__'] = pickle.dumps(model_structure)

    model_name = os.path.basename(model_path).replace('.npy', '')
    output_file = os.path.join(output_dir, f"{model_name}_hash_node.zpkl")

    # 使用临时文件保存，然后移动到输出文件路径
    temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp')
    try:
        with os.fdopen(temp_fd, 'wb') as temp_f:
            # 压缩数据
            compressed_data = zlib.compress(pickle.dumps(processed_model), level=compression_level)
            temp_f.write(compressed_data)

            # 移动文件
        shutil.move(temp_path, output_file)
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

        # 计算文件大小
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Processed model saved for hash tree. File size: {file_size_mb:.2f} MB")

    return output_file, processed_model


def cleanup_temp_files():
    """
    清理所有临时文件
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
    验证生成的哈希节点文件是否具有唯一性
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

                # 检查模型ID唯一性
            model_id = model_data.get('__model_id__', 'unknown')
            if model_id in model_ids:
                print(f"WARNING: Model ID collision between {model_name} and {model_ids[model_id]}")
            else:
                model_ids[model_id] = model_name

                # 检查参数唯一性
            for param_name, param_data in model_data.items():
                if param_name.startswith('__'):  # 跳过元数据
                    continue

                    # 提取参数特征
                try:
                    feature_length = int.from_bytes(param_data[:4], byteorder='big')
                    feature_bytes = param_data[4:4 + feature_length]
                    param_features = pickle.loads(feature_bytes)

                    # 创建参数特征哈希
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

    # 建立输出文件路径
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

        # 打印结果摘要
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

            # 验证生成的哈希节点文件
    verify_hash_node_files(output_dir)

    print("\nAll models have been processed!")


def extract_data_from_hash_node(file_path, param_name=None, include_identifier=True):
    """
    从哈希节点文件中提取参数的加密数据

    参数:
        file_path: 哈希节点文件路径
        param_name: 指定的参数名，如果为None则返回所有参数的加密数据
        include_identifier: 是否在加密数据中包含唯一标识符（默认True）

    返回:
        如果param_name指定: 返回单个参数的特征和加密数据
        如果param_name为None: 返回所有参数的加密数据字典，每个加密数据包含唯一标识
    """
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
        model_data = pickle.loads(zlib.decompress(compressed_data))

    # 如果指定了参数名，只返回该参数的数据
    if param_name is not None:
        if param_name not in model_data:
            print(f"Parameter {param_name} not found in {file_path}")
            return None

        param_data = model_data[param_name]

        # 提取特征和加密数据
        feature_length = int.from_bytes(param_data[:4], byteorder='big')
        feature_bytes = param_data[4:4 + feature_length]
        encrypted_bytes = param_data[4 + feature_length:]

        features = pickle.loads(feature_bytes)

        # 如果需要包含标识符（用于chameleon hash），将部分特征哈希添加到加密数据
        if include_identifier:
            # 创建唯一标识
            param_id = hashlib.sha256(feature_bytes).digest()[:16]  # 使用16字节标识
            encrypted_bytes = param_id + encrypted_bytes

        return {
            'features': features,
            'encrypted_bytes': encrypted_bytes
        }

        # 如果没有指定参数名，返回所有参数的加密数据
    else:
        result = {}
        # 获取模型ID作为额外标识
        model_id = model_data.get('__model_id__', b'').encode()[:8]

        for key, param_data in model_data.items():
            # 跳过元数据字段
            if key.startswith('__'):
                continue

            try:
                # 提取加密数据和特征
                feature_length = int.from_bytes(param_data[:4], byteorder='big')
                feature_bytes = param_data[4:4 + feature_length]
                encrypted_bytes = param_data[4 + feature_length:]

                # 如果需要包含唯一标识，将参数特征哈希添加到加密数据
                if include_identifier:
                    # 为参数创建唯一标识（使用参数名和特征的哈希）
                    key_bytes = key.encode()
                    param_hash = hashlib.sha256(key_bytes + feature_bytes).digest()[:16]

                    # # 组合模型ID、参数ID和加密数据
                    # unique_bytes = model_id + param_hash + encrypted_bytes
                    unique_bytes = param_hash
                    result[key] = unique_bytes
                else:
                    # 仅返回加密数据
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


