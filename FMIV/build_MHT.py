import numpy as np
import hashlib
import time
import json
import sys
import os
import psutil
from typing import List, Dict, Tuple, Any, Optional
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
import random
import pickle


class MHTNode:
    """Merkle Hash Tree节点"""

    def __init__(self):
        """初始化MHT节点"""
        self.hash_value = None  # 节点哈希值
        self.is_leaf = False  # 是否为叶节点
        self.data = None  # 叶节点数据
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.parent = None  # 父节点
        self.secure_code = None
        # 用于可视化和调试
        self.param_id = None  # 参数ID (对叶节点)
        self.model_id = None  # 模型ID (对模型子树根节点)
        self.block_idx = None  # 参数块的索引


class MerkleHashTree:
    """基于SHA-256的Merkle哈希树实现"""

    def __init__(self):
        """初始化Merkle哈希树"""
        self.root = None
        self.model_trees = {}  # {model_id: model_root_node}
        self.model_params = {}  # {model_id: {param_id: [leaf_nodes]}} - 保存每个参数的所有块节点
        self.param_blocks_data = {}  # {model_id: {param_id: [block1, block2, ...]}} - 存储每个参数的分块数据
        self.param_blocks_count = {}  # {model_id: {param_id: num_blocks}} - 存储每个参数的块数
        self.timestamp = None
        self.version = None
        self.node_count = 0
        self.performance_stats = {}

        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def secure_hash(self, data):
        """
        计算数据的安全哈希值

        参数:
            data: 要哈希的数据

        返回:
            str: 数据的十六进制哈希值
        """
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, str):
            data_bytes = data.encode('UTF-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('UTF-8')

            # 初始化SHA-256哈希函数
        sha256 = hashlib.sha256()

        # 更新哈希函数
        sha256.update(data_bytes)

        # 返回十六进制摘要
        return sha256.hexdigest()

    def generate_secure_code(self, f=None):
        """
        生成安全随机码

        参数:
            f: 前一个安全码(可选)

        返回:
            str: 介于1到10000之间的随机整数字符串
        """
        # 生成随机整数
        random_int = random.randint(1, 10000)

        # 转换为字符串并返回
        return str(random_int)

    def encrypt_root_hash(self, root_hash):
        """
        使用RSA加密根哈希值

        参数:
            root_hash: 根哈希值

        返回:
            bytes: 加密后的哈希值
        """
        if isinstance(root_hash, str):
            root_hash_bytes = root_hash.encode('UTF-8')
        elif isinstance(root_hash, bytes):
            root_hash_bytes = root_hash
        else:
            root_hash_bytes = str(root_hash).encode('UTF-8')

            # 使用RSA公钥加密
        encrypted_hash = self.public_key.encrypt(
            root_hash_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return encrypted_hash

    def build_mht(self, data_blocks, f=None):
        """
        构建Merkle哈希树

        参数:
            data_blocks: 数据块列表 [(model_id, param_id, block_idx, data), ...]
            f: 安全码(可选)

        返回:
            MHTNode: 树的根节点
        """
        # 如果没有数据块，返回None
        if not data_blocks:
            return None

            # 如果只有一个数据块，直接创建叶子节点
        if len(data_blocks) == 1:
            model_id, param_id, block_idx, data = data_blocks[0]
            leaf_node = MHTNode()
            leaf_node.is_leaf = True
            leaf_node.data = data
            leaf_node.hash_value = self.secure_hash(data)
            leaf_node.model_id = model_id
            leaf_node.param_id = param_id
            leaf_node.block_idx = block_idx
            return leaf_node

            # 创建所有叶子节点
        leaf_nodes = []
        for model_id, param_id, block_idx, data in data_blocks:
            leaf_node = MHTNode()
            leaf_node.is_leaf = True
            leaf_node.data = data
            leaf_node.hash_value = self.secure_hash(data)
            leaf_node.model_id = model_id
            leaf_node.param_id = param_id
            leaf_node.block_idx = block_idx
            leaf_nodes.append(leaf_node)

            # 递归构建上层节点
        return self._build_tree_from_nodes(leaf_nodes, f)

    def _build_tree_from_nodes(self, nodes, f=None):
        """
        从节点列表构建树

        参数:
            nodes: 节点列表
            f: 安全码

        返回:
            MHTNode: 根节点
        """
        # 基本情况：只有一个节点时返回该节点
        if len(nodes) == 1:
            return nodes[0]

            # 成对处理节点
        parent_nodes = []

        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # 检查是否有右节点
            right_node = nodes[i + 1] if i + 1 < len(nodes) else None

            # 生成安全码
            if f is None:
                secure_code = self.generate_secure_code()
            else:
                secure_code = f

            # 创建父节点
            parent = MHTNode()
            parent.left = left_node
            parent.right = right_node

            # 连接父子关系
            left_node.parent = parent
            if right_node:
                right_node.parent = parent

                # 计算组合哈希值
            if right_node:
                combined_data = str(left_node.hash_value) + str(right_node.hash_value) + secure_code
            else:
                combined_data = str(left_node.hash_value) + secure_code

            parent.hash_value = self.secure_hash(combined_data)
            parent.secure_code = secure_code

            # 添加到父节点列表
            parent_nodes.append(parent)

            # 递归构建上层树
        return self._build_tree_from_nodes(parent_nodes, secure_code)

    def chunk_parameters(self, params_array, chunk_size):
        """
        将参数数组分割成固定大小的块

        参数:
            params_array: 参数数组
            chunk_size: 每个块的大小

        返回:
            chunks: 参数块列表
            time_taken: 分块操作耗时
        """
        start_time = time.time()

        # 分割参数数组
        if isinstance(params_array, np.ndarray):
            # 使用NumPy的数组分割操作
            # 确保使用flatten()将数组展平
            flat_array = params_array.flatten()
            total_elements = len(flat_array)
            num_chunks = int(np.ceil(total_elements / chunk_size))

            chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_elements)
                chunks.append(flat_array[start_idx:end_idx])
        else:
            # 处理非NumPy数组的情况
            chunks = [params_array[i:i + chunk_size] for i in range(0, len(params_array), chunk_size)]

        time_taken = time.time() - start_time

        print(f"参数分块完成，生成 {len(chunks)} 个块，耗时 {time_taken:.4f} 秒")

        return chunks, time_taken

    def determine_chunk_size(self, params_array, target_chunks=16):
        """
        确定参数分块的最佳大小

        参数:
            params_array: 待分块的参数数组
            target_chunks: 目标块数量（默认16块）

        返回:
            chunk_size: 每个块的大小
            num_chunks: 实际的块数量
        """
        if isinstance(params_array, np.ndarray):
            total_elements = params_array.size
        else:
            total_elements = len(params_array)

            # 计算块大小，向上取整确保所有参数都包含在内
        chunk_size = max(1, int(np.ceil(total_elements / target_chunks)))

        # 计算实际块数
        num_chunks = int(np.ceil(total_elements / chunk_size))

        print(f"参数总数: {total_elements}, 目标块数: {target_chunks}")
        print(f"计算的块大小: {chunk_size}, 实际块数: {num_chunks}")

        return chunk_size, num_chunks

    def build_from_model_params(self, all_model_params: Dict[str, Dict[str, np.ndarray]], target_chunks=16):
        """
        从模型参数构建Merkle哈希树并加密根哈希

        参数:
            all_model_params: 字典 {model_id: {param_id: param_data}}
            target_chunks: 每个参数的目标块数

        返回:
            dict: 性能统计字典
        """
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        print(f"构建MHT，包含 {len(all_model_params)} 个模型")

        # 初始化参数块计数和存储
        self.param_blocks_count = {}
        self.param_blocks_data = {}
        self.model_params = {}

        # 转换所有模型参数为叶子数据块
        all_data_blocks = []

        # 首先为每个模型参数分块
        for model_id in all_model_params.keys():
            model_params = all_model_params[model_id]
            self.param_blocks_count[model_id] = {}
            self.param_blocks_data[model_id] = {}
            self.model_params[model_id] = {}

            for param_id in model_params.keys():
                param_data = model_params[param_id]
                self.param_blocks_data[model_id][param_id] = []

                if isinstance(param_data, np.ndarray):
                    # 将大型数组展平后分块
                    flat_params = param_data.flatten()
                    chunk_size, num_chunks = self.determine_chunk_size(flat_params, target_chunks)
                    chunks, _ = self.chunk_parameters(flat_params, chunk_size)

                    # 保存块数量
                    self.param_blocks_count[model_id][param_id] = len(chunks)

                    # 将每个块添加到参数块列表和全局数据块列表
                    for i, chunk in enumerate(chunks):
                        block_data = chunk.tobytes()
                        self.param_blocks_data[model_id][param_id].append(block_data)
                        all_data_blocks.append((model_id, param_id, i, block_data))
                else:
                    # 非数组类型也分块处理
                    data_str = str(param_data)
                    chunk_size, num_chunks = self.determine_chunk_size(data_str, target_chunks)
                    chunks = [data_str[i:i + chunk_size] for i in range(0, len(data_str), chunk_size)]

                    # 保存块数量
                    self.param_blocks_count[model_id][param_id] = len(chunks)

                    # 将每个块添加到参数块列表和全局数据块列表
                    for i, chunk in enumerate(chunks):
                        block_data = chunk.encode('UTF-8')
                        self.param_blocks_data[model_id][param_id].append(block_data)
                        all_data_blocks.append((model_id, param_id, i, block_data))

        print(f"总共 {len(all_data_blocks)} 个数据块准备构建树")

        # 使用安全码生成初始值
        initial_secure_code = self.generate_secure_code()

        # 构建整个树
        self.root = self.build_mht(all_data_blocks, initial_secure_code)

        # 更新模型参数节点映射 - 在树构建后处理
        self._update_model_params_mapping(self.root)

        # 记录时间戳和版本
        self.timestamp = int(time.time())
        self.version = 1

        # 获取根哈希
        root_hash = self.root.hash_value

        # 加密根哈希
        encrypted_hash = self.encrypt_root_hash(root_hash)

        total_time = (time.time() - start_time)
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # 计算存储大小
        tree_size = self._calculate_tree_size()

        print(f"\n树的构建成功！")
        print(f"根哈希值: 0x{root_hash[:16] if isinstance(root_hash, str) else root_hash.hex()[:16]}...")
        print(f"用时: {total_time:.4f}秒")
        print(f"内存使用: {memory_used:.2f}MB")
        print(f"树大小: {tree_size / (1024 * 1024):.2f}MB")

        # 记录性能统计
        performance = {
            "total_blocks": len(all_data_blocks),
            "build_time_sec": total_time,
            "memory_used_mb": memory_used,
            "tree_size_mb": tree_size / (1024 * 1024)
        }

        self.performance_stats["build"] = performance
        return self, performance

    def _update_model_params_mapping(self, node):
        """
        更新模型参数到叶节点的映射

        参数:
            node: 开始搜索的节点
        """
        if node is None:
            return

        if node.is_leaf:
            # 找到叶节点，更新映射
            if node.model_id and node.param_id is not None:
                if node.model_id not in self.model_params:
                    self.model_params[node.model_id] = {}

                if node.param_id not in self.model_params[node.model_id]:
                    self.model_params[node.model_id][node.param_id] = []

                    # 添加叶节点到参数映射
                self.model_params[node.model_id][node.param_id].append(node)
        else:
            # 递归处理子节点
            self._update_model_params_mapping(node.left)
            self._update_model_params_mapping(node.right)

    def _calculate_tree_size(self):
        """计算树的总大小（字节）"""
        if not self.root:
            return 0

        total_size = 0
        visited = set()
        stack = [self.root]

        while stack:
            node = stack.pop()

            if id(node) in visited:
                continue

            visited.add(id(node))

            # 计算节点大小
            node_size = sys.getsizeof(node)

            # 添加哈希值大小
            if node.hash_value:
                node_size += sys.getsizeof(node.hash_value)

                # 添加安全码大小
            if node.secure_code:
                node_size += sys.getsizeof(node.secure_code)

                # 添加数据大小（如果是叶节点）
            if node.is_leaf and node.data:
                if isinstance(node.data, np.ndarray):
                    node_size += node.data.nbytes
                else:
                    node_size += sys.getsizeof(node.data)

            total_size += node_size

            # 添加子节点
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return total_size

    def save_to_file(self, output_file, include_public_key=True):
        """
        将MHT的元数据保存到文件

        参数:
            output_file: 输出文件路径
            include_public_key: 是否包含公钥
        """
        if not self.root:
            print("错误：树为空，无法保存")
            return

            # 将根哈希加密
        encrypted_hash = self.encrypt_root_hash(self.root.hash_value)

        # 创建可序列化的树表示
        tree_data = {
            'timestamp': self.timestamp,
            'version': self.version,
            'root_hash': self.root.hash_value if isinstance(self.root.hash_value, str) else self.root.hash_value.hex(),
            'encrypted_root_hash': encrypted_hash.hex(),
            'performance_stats': self.performance_stats,
            'param_blocks_count': self.param_blocks_count
        }

        # 添加公钥（如果需要）
        if include_public_key:
            tree_data['public_key_pem'] = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')

        with open(output_file, 'w') as f:
            json.dump(tree_data, f, indent=2)

        print(f"MHT元数据已保存到 {output_file}")

    def get_model_proof(self, model_id_str: str) -> Dict[str, Any]:
        """
        获取模型的证明路径
        """
        if model_id_str not in self.model_params:
            raise ValueError(f"模型 {model_id_str} 不存在")

        model_params = self.model_params[model_id_str]
        blocks_data = self.param_blocks_data[model_id_str]

        # 构建模型参数证明
        params_data = {}
        params_proofs = {}

        for param_id, leaf_nodes in model_params.items():
            # 对叶节点按块索引排序
            sorted_nodes = sorted(leaf_nodes, key=lambda x: x.block_idx)

            # 为每个块构建证明路径
            blocks_proofs = []
            for node in sorted_nodes:
                # 获取从叶节点到根的证明路径
                proof_path = []
                current = node

                while current.parent is not None:
                    is_left = current == current.parent.left
                    sibling = current.parent.right if is_left else current.parent.left

                    if sibling:
                        proof_path.append({
                            'position': 'left' if not is_left else 'right',
                            'hash': sibling.hash_value,
                            'secure_code': current.parent.secure_code
                        })

                    current = current.parent

                blocks_proofs.append({
                    'block_idx': node.block_idx,
                    'proof': proof_path
                })

                # 将块排序并保存参数数据
            params_data[param_id] = blocks_data[param_id]

            # 保存参数的所有块证明路径
            params_proofs[param_id] = blocks_proofs

        return {
            'model_id': model_id_str,
            'params': params_data,  # 这是一个字典，键为param_id，值为块列表
            'params_proofs': params_proofs,
            'global_root_hash': self.root.hash_value,
            'timestamp': self.timestamp,
            'version': self.version,
            'param_blocks_count': self.param_blocks_count[model_id_str]
        }

    def verify_model_proof(self, proof: Dict[str, Any]) -> bool:
        """
        验证模型证明

        参数:
            proof: 通过get_model_proof获取的证明

        返回:
            bool: 验证是否成功
        """
        # 提取证明信息
        model_id = proof['model_id']
        global_root_hash = proof['global_root_hash']
        params = proof['params']  # {param_id: [block1, block2, ...]}
        params_proofs = proof['params_proofs']
        param_blocks_count = proof.get('param_blocks_count', {})

        # 验证每个参数
        for param_id, param_blocks in params.items():
            if param_id not in params_proofs:
                print(f"参数 {param_id} 的证明缺失")
                return False

            blocks_proofs = params_proofs[param_id]

            # 验证块数量
            if param_id in param_blocks_count and len(blocks_proofs) != param_blocks_count[param_id]:
                print(f"参数 {param_id} 的块数不匹配: 期望 {param_blocks_count[param_id]}, 实际 {len(blocks_proofs)}")
                return False

                # 验证每个块
            for proof_item in blocks_proofs:
                block_idx = proof_item['block_idx']

                if block_idx >= len(param_blocks):
                    print(f"块索引 {block_idx} 超出范围")
                    return False

                block_data = param_blocks[block_idx]

                # 计算叶节点哈希
                leaf_hash = self.secure_hash(block_data)

                # 按照证明路径逐步验证
                current_hash = leaf_hash
                for step in proof_item['proof']:
                    sibling_hash = step['hash']
                    secure_code = step.get('secure_code', '')

                    if step['position'] == 'left':
                        # 当前节点在右侧
                        combined = sibling_hash + current_hash + secure_code
                    else:
                        # 当前节点在左侧
                        combined = current_hash + sibling_hash + secure_code

                    current_hash = self.secure_hash(combined)

                    # 验证是否最终得到全局根哈希
                if current_hash != global_root_hash:
                    print(f"参数 {param_id} 块 {block_idx} 证明验证失败")
                    return False

        return True

def save_merkle_hash_tree(model_tree, filepath):
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        # 使用pickle序列化对象并保存
        with open(filepath, 'wb') as f:
            pickle.dump(model_tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"ChameleonHashTree成功保存到: {filepath}")
        return True
    except Exception as e:
        print(f"保存ChameleonHashTree时出错: {e}")
        return False


def load_merkle_hash_tree(filepath):
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在 {filepath}")
        return None

    try:
        # 使用pickle加载对象
        with open(filepath, 'rb') as f:
            model_tree = pickle.load(f)
        print(f"ChameleonHashTree成功从 {filepath} 加载")
        return model_tree
    except Exception as e:
        print(f"加载ChameleonHashTree时出错: {e}")
        return None


def get_MHT_multi_model():
    all_models_data = {}
    model_id_mapping = {}
    # get model id
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/FMIV/model_id_pre_trained_model.txt", 'r',
              encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    print(model_id_mapping)
    # # get encrypted model params
    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}

        # 加载加密模型参数
        encrypted_data = np.load(encrypted_path, allow_pickle=True)

        # 处理不同类型的NumPy数组
        if isinstance(encrypted_data, np.ndarray) and encrypted_data.dtype == np.dtype('O'):
            # 处理对象数组
            if encrypted_data.ndim == 0:
                # 0维对象数组 - 使用item()获取其中的字典
                model_params = encrypted_data.item()
                if not isinstance(model_params, dict):
                    print(f"警告: 模型 {model_id} 的数据不是字典格式")
                    model_params = {"parameters": model_params}
            else:
                # 多维对象数组 - 通常是数组的第一个元素
                if len(encrypted_data) > 0 and isinstance(encrypted_data[0], dict):
                    model_params = encrypted_data[0]
                else:
                    print(f"警告: 模型 {model_id} 的数据格式不是预期的字典数组")
                    model_params = {"full_array": encrypted_data}
        else:
            # 不是对象数组，可能是直接的数值数组
            print(f"模型 {model_id} 的数据是简单数组格式")
            model_params = {"parameters": encrypted_data}

            # 将参数添加到所有模型数据中
        print(f"处理模型 {model_id}, 参数数量: {len(model_params)}")
        for name, param in model_params.items():
            all_models_data[model_id][name] = param
            if isinstance(param, np.ndarray):
                print(f"  参数 {name}: 形状 {param.shape}, 类型 {param.dtype}")

    mht_builder = MerkleHashTree()
    # 增加traget_chunks可以增加树的构建开销
    MHT, performance = mht_builder.build_from_model_params(all_models_data, target_chunks=18024)

    # 保存MHT元数据
    mht_builder.save_to_file("mht_metadata.json")

    save_merkle_hash_tree(MHT, "./MHT_8.tree")

    return performance


if __name__ == "__main__":
    performance = get_MHT_multi_model()
    print(performance)