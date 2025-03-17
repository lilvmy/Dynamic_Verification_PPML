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
from initialization.setup import load_HE_keys


def get_memory_usage():
    """返回当前内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def binary_mean_representation(param):
    """
    将参数转换为二值表示并计算平均值

    Args:
        param: 原始参数数组

    Returns:
        一个字典，包含二值化信息和平均值
    """
    # 二值化 (>0 为 1, 其他为 0)
    binary = (param > 0).astype(np.float32)

    # 计算二值化后的平均值
    binary_mean = np.mean(binary)

    # 计算原始正值和负值的平均值，用于重建参数
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


def encrypt_model_as_binary_means_dict(model_path, output_dir, compression_level=9):
    """
    将模型参数转换为二值表示，计算平均值，加密并保存为字典结构的单个文件

    Args:
        model_path: 模型参数路径
        output_dir: 输出目录
        compression_level: zlib压缩级别

    Returns:
        输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载加密密钥
    print("Loading HE keys...")
    HE = load_HE_keys()

    # 加载模型
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = np.load(f, allow_pickle=True).item()

    param_names = list(model.keys())
    print(f"Found {len(param_names)} parameters")

    # 创建结果字典 - 这将是我们最终保存的格式
    encrypted_model = {}

    # 处理每个参数
    for name in param_names:
        print(f"Processing parameter: {name}")
        param = model[name]

        # 获取二值表示及平均值
        binary_info = binary_mean_representation(param)
        binary_mean = binary_info['binary_mean']

        print(f"  Shape: {binary_info['shape']}, Elements: {param.size}")
        print(f"  Binary mean: {binary_mean:.4f}")
        print(f"  Positive elements mean: {binary_info['positive_mean']:.4f}")
        print(f"  Negative elements mean: {binary_info['negative_mean']:.4f}")

        # 加密二值平均值
        encrypted_mean = HE.encode_number(binary_mean)

        # 将加密信息存储在结果字典中
        encrypted_model[name] = encrypted_mean.to_bytes()

    # 获取模型名称
    model_name = os.path.basename(model_path).replace('.npy', '')
    output_file = os.path.join(output_dir, f"{model_name}_binary_means.enpy")

    # 使用临时文件保存，然后移动到最终位置
    temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp')
    try:
        with os.fdopen(temp_fd, 'wb') as temp_f:
            # 压缩并保存字典
            compressed_data = zlib.compress(pickle.dumps(encrypted_model), level=compression_level)
            temp_f.write(compressed_data)

        # 将临时文件移到最终位置
        shutil.move(temp_path, output_file)
    except Exception as e:
        # 确保出错时也删除临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

    # 获取文件大小
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Encrypted model saved as dictionary. File size: {file_size_mb:.2f} MB")

    return output_file


def load_encrypted_model_dict(encrypted_file_path):
    """
    加载保存为字典结构的加密模型

    Args:
        encrypted_file_path: 加密模型文件路径

    Returns:
        加载的模型字典
    """
    # 读取加密文件
    with open(encrypted_file_path, 'rb') as f:
        compressed_data = f.read()

    # 解压缩并返回字典
    return pickle.loads(zlib.decompress(compressed_data))


def decrypt_binary_mean_model_dict(encrypted_model_dict):
    """
    解密和重建基于字典结构的二值化平均值模型

    Args:
        encrypted_model_dict: 加密模型字典

    Returns:
        解密后的模型参数字典
    """
    # 加载密钥
    HE = load_HE_keys()

    # 解密并重建参数
    decrypted_model = {}

    for name, param_data in encrypted_model_dict.items():
        # 解密平均值
        encrypted_mean = HE.from_bytes(param_data['encrypted_mean'])
        binary_mean = HE.decode_number(encrypted_mean)

        # 获取参数形状和重建值
        shape = param_data['shape']
        positive_mean = param_data['positive_mean']
        negative_mean = param_data['negative_mean']

        # 使用二值平均值重建二值掩码
        # 这只是一个近似，无法精确重建原始的二值表示
        total_elements = np.prod(shape)
        ones_count = int(binary_mean * total_elements + 0.5)  # 四舍五入

        # 创建扁平化的二值掩码
        binary_flat = np.zeros(total_elements, dtype=np.float32)
        binary_flat[:ones_count] = 1.0

        # 随机排列以避免所有1都在前面
        np.random.shuffle(binary_flat)

        # 重塑为原始形状
        binary = binary_flat.reshape(shape)

        # 根据二值掩码和正/负平均值重建参数
        reconstructed = np.where(binary > 0.5, positive_mean, negative_mean).astype(np.float32)

        decrypted_model[name] = reconstructed

        print(f"  Decrypted {name}: shape {shape}, binary mean {binary_mean:.4f}")

    return decrypted_model


def cleanup_temp_files():
    """清理所有临时文件"""
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        if filepath.endswith('.tmp'):
            try:
                os.unlink(filepath)
            except Exception:
                pass
    gc.collect()


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

    # 创建输出目录
    output_dir = "./encrypted_model_params_binary"
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
            # 加密并保存模型为字典结构
            output_file = encrypt_model_as_binary_means_dict(
                model_path,
                output_dir,
                compression_level=9
            )

            # 记录结果
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

        # 释放内存并清理临时文件
        cleanup_temp_files()
        gc.collect()

    # 打印结果摘要
    print("\n=== Processing Summary ===")
    for model, result in results.items():
        status = result['status']
        if status == 'success':
            print(f"{model}: Successfully encrypted as dictionary")
            print(f"  Output file: {result['output_file']}")
            print(f"  File size: {result['file_size_mb']:.2f} MB")
        elif status == 'failed':
            print(f"{model}: Failed - {result.get('error', 'Unknown error')}")
        else:
            print(f"{model}: Not found")

    print("\nAll models have been processed!")


if __name__ == "__main__":
    # try:
    #     main()
    # finally:
    #     # 确保在程序结束时清理所有临时文件
    #     cleanup_temp_files()


    # # 调试代码
    # 获取加密参数
    loaded_data = load_encrypted_model_dict("./encrypted_model_params_binary/lenet1_model_params_binary_means.enpy")

    # 打印键值信息
    print(f"\n字典中的键数量: {len(loaded_data)}")
    for i, (key, value) in enumerate(loaded_data.items()):
        print(f"键: {key}")

        if isinstance(value, bytes):
            print(f"值类型: bytes (长度: {len(value)} 字节)")
            # 只显示前50个字符的十六进制表示
            hex_preview = value.hex()[:100]
            print(f"十六进制预览: {hex_preview}..." if len(value.hex()) > 100 else f"十六进制: {hex_preview}")
        elif isinstance(value, dict):
            print(f"值类型: 嵌套字典，包含键: {list(value.keys())}")
        else:
            print(f"值类型: {type(value)}")
