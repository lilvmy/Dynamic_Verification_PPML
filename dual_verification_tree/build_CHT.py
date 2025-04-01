import hashlib
import time
from typing import List, Dict, Tuple, Any, Optional
import ecdsa  # ecdsa lib
from initialization.setup import load_ecdsa_keys
from dual_verification_tree.CHT_utils import ChameleonHash, load_cht_keys
from level_homomorphic_encryption.encrypted_process_model import extract_data_from_hash_node
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np
import copy
import traceback

class PublicKeySet:
    """
    storage public key of chameleon hash
    """

    def __init__(self, p, q, g, pk):
        self.p = p  # security prime
        self.q = q  # prime q, p = 2q + 1
        self.g = g  # generator
        self.pk = pk  # public key

    def get_p(self):
        return self.p

    def get_q(self):
        return self.q

    def get_g(self):
        return self.g

    def get_public_key(self):
        return self.pk


class PrivateKeySet(PublicKeySet):
    """
    storage private key of chameleon hash and relevant parameters
    """

    def __init__(self, p, q, g, sk, pk):
        super().__init__(p, q, g, pk)
        self.sk = sk  # private key

    def get_secret_key(self):
        return self.sk

    def get_public_key_set(self):
        return PublicKeySet(self.p, self.q, self.g, self.pk)


class CHTNode:
    """
    node of CHT
    """

    def __init__(self):
        self.hash_value = None  # hash value of node（bytes）
        self.rho = None  # random number rho
        self.delta = None  # random number delta
        self.left = None  # left sub-node
        self.right = None  # right sub-node
        self.parent = None  # parent node
        self.is_leaf = False  # is whether leaf node
        self.data = None  # cite of leaf node data

    def size_in_bytes(self):
        """计算CHTNode节点的内存占用（字节）"""
        # 基本对象开销
        size = sys.getsizeof(self)

        # 计算hash_value的大小
        if self.hash_value is not None:
            size += sys.getsizeof(self.hash_value)

            # 计算rho的大小
        if self.rho is not None:
            size += sys.getsizeof(self.rho)

            # 计算delta的大小
        if self.delta is not None:
            size += sys.getsizeof(self.delta)

        # 注意：我们不计算left, right和parent的大小
        # 因为这些是对其他节点的引用，计算它们会导致重复计算
        # 只计算引用本身的大小
        size += sys.getsizeof(self.left)
        size += sys.getsizeof(self.right)
        size += sys.getsizeof(self.parent)

        # 计算is_leaf的大小
        size += sys.getsizeof(self.is_leaf)

        # 计算data的大小
        if self.data is not None:
            if isinstance(self.data, np.ndarray):
                size += self.data.nbytes
            elif hasattr(self.data, 'size_in_bytes') and callable(getattr(self.data, 'size_in_bytes')):
                # 如果data对象自己有size_in_bytes方法
                size += self.data.size_in_bytes()
            else:
                size += sys.getsizeof(self.data)

        return size

    # ====================== CHT ======================
class ChameleonHashTree:
    """CHT based on discrete log"""

    def __init__(self, keys: PrivateKeySet, security_param: int = 512):
        """
        initialize CHT
        """
        self.keys = keys
        self.security_param = security_param
        self.public_keys = keys.get_public_key_set()
        self.root = None
        self.model_trees = {}  #  {model_id: model_root_node}
        self.model_params = {}  # {model_id: {param_id: leaf_node}}
        self.signature = None  # sign root node
        self.timestamp = None
        self.version = None
        self.node_count = 0

    def calculate_storage_size(self, include_values=True):
        """
        计算整个CHT树的存储大小（字节）

        参数:
            include_values (bool): 是否包括节点中的数据值(hash_value, rho, delta, data)在计算中

        返回:
            int: 存储大小（字节）
        """
        if self.root is None:
            return 0

            # 使用集合来跟踪已访问的节点，避免重复计算
        visited = set()
        total_size = 0

        # 使用栈进行遍历，避免递归导致的栈溢出
        stack = [self.root]

        while stack:
            node = stack.pop()

            # 如果节点已访问，跳过
            if id(node) in visited:
                continue

                # 标记为已访问
            visited.add(id(node))

            # 计算当前节点大小
            if include_values:
                # 计算对象基本大小
                node_size = sys.getsizeof(node)

                # 计算各属性大小
                if node.hash_value is not None:
                    node_size += sys.getsizeof(node.hash_value)

                if node.rho is not None:
                    node_size += sys.getsizeof(node.rho)

                if node.delta is not None:
                    node_size += sys.getsizeof(node.delta)

                    # is_leaf 布尔值的大小已包含在对象基本大小中

                # 计算data的大小
                if node.data is not None:
                    if isinstance(node.data, np.ndarray):
                        node_size += node.data.nbytes
                    elif hasattr(node.data, 'size_in_bytes') and callable(getattr(node.data, 'size_in_bytes')):
                        # 如果data有size_in_bytes方法
                        node_size += node.data.size_in_bytes()
                    else:
                        node_size += sys.getsizeof(node.data)
            else:
                # 只计算节点结构的开销，不包括值
                node_size = sys.getsizeof(node)

                # 累加到总大小
            total_size += node_size

            # 将子节点添加到栈中
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        return total_size

    def _generate_node_name(self, node_type="internal", model_id=None, param_id=None):
        """生成唯一的节点名称用于可视化"""
        self.node_count += 1
        if node_type == "leaf" and param_id:
            short_param = param_id.split('.')[-1][:5]  # 简化参数名
            return f"L{self.node_count}:{short_param}"
        elif node_type == "model" and model_id:
            short_model = model_id[:3]  # 简化模型ID
            return f"M{self.node_count}:{short_model}"
        else:
            return f"N{self.node_count}"

    def build_from_model_params(self, all_model_params: Dict[str, Dict[str, bytes]],signing_key: ecdsa.SigningKey) -> CHTNode:
        """
        build CHT from model parameters
        """
        print(f"build CHT, including {len(all_model_params)} model")

        # build sub-tree for each model
        model_roots = []
        for model_id_str in all_model_params.keys():
            print(f"    build model {model_id_str} sub tree:")
            model_params = all_model_params[model_id_str]

            # generate leaf node of model
            leaf_nodes = []
            param_map = {}

            for param_id_str in model_params.keys():
                node = CHTNode()
                node.is_leaf = True
                node.data = model_params[param_id_str]
                # 存储参数ID，供可视化使用
                node.param_id = param_id_str  # 添加这行

                # encode param info to compute hash
                encoded_data = self._encode_param(model_id_str, param_id_str, model_params[param_id_str])

                # generate ramdon number
                node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                # compute hash value of leaf node
                node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                leaf_nodes.append(node)
                param_map[param_id_str] = node
                hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])
                print(f"  parameter {param_id_str} leaf node：hash value = 0x{hash_str}...")

            model_root = self._build_internal_nodes(leaf_nodes)
            # 存储模型ID，供可视化使用
            model_root.model_id = model_id_str[0:4]  # 添加这行
            model_roots.append(model_root)

            # save model sub tree and param mapping
            self.model_trees[model_id_str] = model_root
            self.model_params[model_id_str] = param_map

            hash_str = ''.join(f'{b:02x}' for b in model_root.hash_value[:4])
            print(f"  model {model_id_str} sub-tree building successfully, root hash: 0x{hash_str}...")

        self.root = self._build_internal_nodes(model_roots)

        # sign for root node
        self.timestamp = int(time.time())
        self.version = 1
        root_hash_hex = ''.join(f'{b:02x}' for b in self.root.hash_value)
        message = f"{root_hash_hex}|{self.timestamp}|{self.version}".encode()
        self.signature = signing_key.sign(message, hashfunc=hashlib.sha256)

        print(
            f"global tree building successfully, root hash: 0x{''.join(f'{b:02x}' for b in self.root.hash_value[:8])}...")

        return self.root

    def _encode_param(self, model_id: str, param_id: str, data: bytes) -> bytes:
        """
        encode model param to compute hash
        """
        model_bytes = model_id.encode('utf-8')
        param_bytes = param_id.encode('utf-8')

        # add length prefix to ensure unique decoding
        model_len = len(model_bytes).to_bytes(2, byteorder='big')
        param_len = len(param_bytes).to_bytes(2, byteorder='big')

        return model_len + model_bytes + param_len + param_bytes + data

    def _build_internal_nodes(self, nodes: List[CHTNode]) -> CHTNode:
        """
        recursive build internal node
        """
        if len(nodes) == 1:
            return nodes[0]

        parent_nodes = []

        # create parent nodes in pairs
        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # if right node exists
            if i + 1 < len(nodes):
                right_node = nodes[i + 1]

                # build parent node
                parent = CHTNode()
                parent.left = left_node
                parent.right = right_node
                left_node.parent = parent
                right_node.parent = parent

                # the hash value which join left node and right node
                combined_data = left_node.hash_value + right_node.hash_value

                parent.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                parent.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                parent.hash_value = ChameleonHash.hash(combined_data, parent.rho, parent.delta, self.public_keys)

                parent_nodes.append(parent)
            else:
                # when there are an odd number of nodes, directly promote the last node
                parent_nodes.append(left_node)

                # recursively build upper nodes
        return self._build_internal_nodes(parent_nodes)

    def get_model_proof(self, model_id_str: str) -> Dict[str, Any]:
        """
        get a proof path for a model
        """
        if model_id_str not in self.model_trees:
            raise ValueError(f"model {model_id_str} is none")

        model_root = self.model_trees[model_id_str]
        model_params = self.model_params[model_id_str]

        # build the proof path from sub tree root to global root
        proof_path = []
        current = model_root

        while current.parent is not None:
            is_left = current == current.parent.left

            sibling = current.parent.right if is_left else current.parent.left

            if sibling:
                proof_path.append({
                    'position': 'left' if not is_left else 'right',
                    'hash': sibling.hash_value,
                    'rho': current.parent.rho,
                    'delta': current.parent.delta
                })

            current = current.parent

        # build verification path for model
        params_data = {}
        params_proofs = {}

        for param_id_str, leaf_node in model_params.items():
            # get param
            params_data[param_id_str] = leaf_node.data

            # get proof path form leaf node to sub tree root
            param_proof = []
            current = leaf_node

            while current.parent is not None and current != model_root:
                is_left = current == current.parent.left

                sibling = current.parent.right if is_left else current.parent.left

                if sibling:
                    param_proof.append({
                        'position': 'left' if not is_left else 'right',
                        'hash': sibling.hash_value,
                        'rho': current.parent.rho,
                        'delta': current.parent.delta
                    })

                current = current.parent

            # save proof path of param
            params_proofs[param_id_str] = {
                'rho': leaf_node.rho,
                'delta': leaf_node.delta,
                'proof': param_proof
            }

        return {
            'model_id': model_id_str,
            'params': params_data,
            'params_proofs': params_proofs,
            'model_root_hash': model_root.hash_value,
            'global_proof': proof_path,
            'global_root_hash': self.root.hash_value,
            'timestamp': self.timestamp,
            'version': self.version,
            'signature': self.signature
        }

    def update_model_or_params(self,model_to_add: Dict[str, Dict[str, bytes]] = None,model_id_to_delete: str = None,param_modifications: Dict[str, Dict[str, bytes]] = None) -> CHTNode:
        """更新模型或参数（增加模型、删除模型、修改参数）并维持CHT完整性

        Args:
            model_to_add: 要添加的新模型 {model_id: {param_id: data, ...}}
            model_id_to_delete: 要删除的模型ID
            param_modifications: 要修改的参数 {model_id: {param_id: new_data, ...}}

        Returns:
            更新后的树根节点
        """
        # 跟踪修改的路径
        modified_paths = []

        # 1. 处理删除整个模型
        if model_id_to_delete:
            if model_id_to_delete not in self.model_trees:
                print(f"模型 {model_id_to_delete} 不存在，无法删除")
            else:
                try:
                    # 获取要删除的模型子树根节点
                    model_root = self.model_trees[model_id_to_delete]

                    # 我们不能实际从树中删除节点，但可以用空模型替换它
                    # 创建一个包含单个空参数的空模型
                    empty_param_id = "_empty_"
                    empty_param_data = b''

                    # 创建空模型的叶节点
                    node = CHTNode()
                    node.is_leaf = True
                    node.data = empty_param_data
                    node.param_id = empty_param_id  # 为可视化添加参数ID

                    # 编码参数信息
                    encoded_data = self._encode_param(model_id_to_delete, empty_param_id, empty_param_data)

                    # 生成随机数
                    node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                    node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                    # 计算叶节点哈希
                    node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                    # 这个单一节点现在代表整个被删除的模型
                    new_model_root = node
                    new_model_root.model_id = model_id_to_delete[0:4]  # 为可视化添加模型ID

                    # 更新从模型根到全局根的路径
                    current_node = model_root
                    new_node = new_model_root

                    # 保存修改路径信息
                    path = []

                    # 递归地向上更新父节点，直到到达全局根
                    while current_node.parent is not None:
                        parent = current_node.parent

                        # 记录修改路径
                        parent_hash_before = parent.hash_value

                        # 确定当前节点是左节点还是右节点
                        is_left = current_node == parent.left

                        # 获取兄弟节点
                        sibling = parent.right if is_left else parent.left

                        # 创建新的编码数据（左右子节点的哈希连接）
                        if is_left:
                            combined_data = new_node.hash_value + sibling.hash_value
                        else:
                            combined_data = sibling.hash_value + new_node.hash_value

                            # 寻找新数据的哈希碰撞
                        pre_image = ChameleonHash.forge(parent.hash_value, combined_data, self.keys)

                        # 如果是左节点，更新父节点的左子节点，否则更新右子节点
                        if is_left:
                            parent.left = new_node
                        else:
                            parent.right = new_node

                        new_node.parent = parent

                        # 更新父节点的rho和delta
                        parent.rho = pre_image.rho
                        parent.delta = pre_image.delta

                        # 验证哈希值保持不变
                        new_hash = ChameleonHash.hash(combined_data, pre_image.rho, pre_image.delta, self.public_keys)
                        if new_hash != parent.hash_value:
                            raise Exception("更新树结构时哈希碰撞验证失败")

                            # 记录节点修改前后的信息
                        path.append({
                            'node_level': 'internal',
                            'original_hash': parent_hash_before.hex(),
                            'new_hash': parent.hash_value.hex(),
                            'position': 'left' if is_left else 'right',
                            'rho': pre_image.rho,
                            'delta': pre_image.delta
                        })

                        # 移动到上一级继续处理
                        current_node = parent
                        new_node = parent

                        # 更新模型映射
                    # 创建新的空参数映射
                    empty_params = {empty_param_id: node}

                    # 更新模型树和参数映射
                    self.model_trees[model_id_to_delete] = new_model_root
                    self.model_params[model_id_to_delete] = empty_params


                    print(f"模型 {model_id_to_delete} 已成功标记为删除，保持全局树结构不变")

                except Exception as e:
                    print(f"删除模型 {model_id_to_delete} 失败: {str(e)}")

        # 2. 处理修改模型参数
        if param_modifications:
            # 首先更新叶子节点的数据
            modified_model_ids = set()
            for model_id, params in param_modifications.items():
                if model_id not in self.model_params:
                    for param_id in params:
                        print(f"修改参数失败: 模型 {model_id} 不存在")
                    continue

                model_params = self.model_params[model_id]
                model_modified = False

                for param_id, new_data in params.items():
                    if param_id not in model_params:
                        print(f"修改参数失败: 参数 {param_id} 不存在于模型 {model_id} 中")
                        continue

                    try:
                        # 更新叶子节点数据
                        leaf_node = model_params[param_id]
                        leaf_node.data = new_data
                        model_modified = True
                        print(f"参数(模型{model_id}参数{param_id})数据已更新，准备重建树")

                    except Exception as e:
                        print(f"修改参数(模型{model_id}参数{param_id})失败: {str(e)}")

                        # 如果模型有修改，记录下来
                if model_modified:
                    modified_model_ids.add(model_id)

                    # 开始重建树
            print(f"开始重建树结构，涉及 {len(modified_model_ids)} 个修改的模型...")

            try:
                # 重建所有修改过的模型子树
                for model_id in modified_model_ids:
                    model_params_dict = {}
                    # 收集模型的所有参数
                    for param_id, leaf_node in self.model_params[model_id].items():
                        model_params_dict[param_id] = leaf_node.data

                        # 清除旧的模型子树
                    old_model_root = self.model_trees[model_id]

                    # 为修改后的模型创建新的叶子节点和子树
                    leaf_nodes = []
                    param_map = {}

                    for param_id_str in model_params_dict.keys():
                        node = CHTNode()
                        node.is_leaf = True
                        node.data = model_params_dict[param_id_str]
                        node.param_id = param_id_str

                        # 编码参数数据
                        encoded_data = self._encode_param(model_id, param_id_str, model_params_dict[param_id_str])

                        # 生成新的哈希参数
                        node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                        node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                        # 计算新的哈希值
                        node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                        leaf_nodes.append(node)
                        param_map[param_id_str] = node
                        hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])
                        print(f"  重建参数 {param_id_str} 叶子节点：哈希值 = 0x{hash_str}...")

                        # 构建模型子树
                    new_model_root = self._build_internal_nodes(leaf_nodes)
                    new_model_root.model_id = model_id[0:8]

                    # 更新模型树和参数映射
                    self.model_trees[model_id] = new_model_root
                    self.model_params[model_id] = param_map

                    hash_str = ''.join(f'{b:02x}' for b in new_model_root.hash_value[:4])
                    print(f"  模型 {model_id} 子树重建成功，根哈希: 0x{hash_str}...")

                    # 收集所有模型的子树根节点
                all_model_roots = list(self.model_trees.values())

                # 保存当前全局根哈希
                old_root_hash = self.root.hash_value.hex() if self.root else None

                # 重建全局树
                new_global_root = self._build_internal_nodes(all_model_roots)

                # 更新全局根节点
                self.root = new_global_root

                # 记录新根哈希
                new_root_hash = self.root.hash_value.hex()

                print(f"全局根哈希从 {old_root_hash[:8]}... 变更为 {new_root_hash[:8]}...")

            except Exception as e:
                print(f"重建树结构失败: {str(e)}")
                traceback.print_exc()

            print(
                f"总共修改了 {sum(len(params) for model_id, params in param_modifications.items())} 个参数")

            # 返回更新后的树
            return self

            # 3. 处理添加整个模型
        if model_to_add:
            try:
                # 为新模型构建子树
                new_model_roots = []
                for model_id_str in model_to_add.keys():
                    model_params = model_to_add[model_id_str]

                    leaf_nodes = []
                    param_map = {}

                    for param_id_str in model_params.keys():
                        node = CHTNode()
                        node.is_leaf = True
                        node.data = model_params[param_id_str]
                        node.param_id = param_id_str

                        encoded_data = self._encode_param(model_id_str, param_id_str, model_params[param_id_str])

                        node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                        node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                        node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                        leaf_nodes.append(node)
                        param_map[param_id_str] = node
                        hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])
                        print(f"  parameter {param_id_str} leaf node：hash value = 0x{hash_str}...")

                    new_model_root = self._build_internal_nodes(leaf_nodes)

                    new_model_root.model_id = model_id_str[0:4]
                    new_model_roots.append(new_model_root)

                    self.model_trees[model_id_str] = new_model_root
                    self.model_params[model_id_str] = param_map

                    hash_str = ''.join(f'{b:02x}' for b in new_model_root.hash_value[:4])
                    print(f"  model {model_id_str} sub-tree building successfully, root hash: 0x{hash_str}...")

                self.root = self._build_internal_nodes(new_model_roots)

                # 首先，获取当前所有模型的子树根节点
                current_model_roots = list(self.model_trees.values())

                # 合并当前模型根和新模型根
                all_model_roots = current_model_roots

                # 保存当前全局根哈希
                old_root_hash = self.root.hash_value.hex() if self.root else None

                # 重建全局树
                new_global_root = self._build_internal_nodes(all_model_roots)

                # 更新全局根节点
                self.root = new_global_root

                # 记录新根哈希
                new_root_hash = self.root.hash_value.hex()

            except Exception as e:
                print(f"添加模型失败: {str(e)}")

                # 返回树结构
        return self

def draw_tree(tree_or_root, output_file="./figure/CHT.png", max_depth=15):
    """
    绘制变色龙哈希树，使用优化的布局算法，自适应节点间距，并简化节点标签

    参数:
        tree_or_root: 可以是ChameleonHashTree对象或CHTNode对象
        output_file: 输出文件路径
        max_depth: 最大显示深度，设置较大值以显示完整树
    """
    # 设置matplotlib参数确保字体嵌入和高质量输出
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42  # 使用TrueType字体
    mpl.rcParams['ps.fonttype'] = 42  # 使用TrueType字体
    mpl.rcParams['svg.fonttype'] = 'none'  # 使用SVG原生文本
    mpl.rcParams['figure.dpi'] = 150  # 默认DPI

    # 使用标准无衬线字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

    # 确定根节点
    if isinstance(tree_or_root, ChameleonHashTree):
        root = tree_or_root.root
        model_trees = tree_or_root.model_trees if hasattr(tree_or_root, 'model_trees') else {}
        model_params = tree_or_root.model_params if hasattr(tree_or_root, 'model_params') else {}
    else:
        root = tree_or_root
        model_trees = {}
        model_params = {}

    if root is None:
        print("错误：根节点为空，无法绘制树")
        return

        # 创建反向映射：从节点到模型ID和参数ID
    node_to_model_id = {}
    node_to_param_id = {}

    # 填充映射信息
    for model_id, model_root in model_trees.items():
        node_to_model_id[id(model_root)] = model_id

        if model_id in model_params:
            for param_id, param_node in model_params[model_id].items():
                node_to_param_id[id(param_node)] = param_id

                # 收集树的层次和叶子节点

    def analyze_tree(node, level=0):
        if node is None:
            return [], {}, [], 0

            # 处理当前节点
        all_nodes = [(node, level)]
        level_map = {id(node): level}
        max_lvl = level

        # 收集叶子节点
        leaf_nodes = []
        if hasattr(node, 'is_leaf') and node.is_leaf:
            leaf_nodes.append(node)

            # 处理左子树
        if hasattr(node, 'left') and node.left:
            left_nodes, left_level_map, left_leaves, left_max = analyze_tree(node.left, level + 1)
            all_nodes.extend(left_nodes)
            level_map.update(left_level_map)
            leaf_nodes.extend(left_leaves)
            max_lvl = max(max_lvl, left_max)

            # 处理右子树
        if hasattr(node, 'right') and node.right:
            right_nodes, right_level_map, right_leaves, right_max = analyze_tree(node.right, level + 1)
            all_nodes.extend(right_nodes)
            level_map.update(right_level_map)
            leaf_nodes.extend(right_leaves)
            max_lvl = max(max_lvl, right_max)

        return all_nodes, level_map, leaf_nodes, max_lvl

        # 分析树结构

    all_nodes, level_map, leaf_nodes, max_level = analyze_tree(root)

    # 确保叶子节点都在最底层
    for node in leaf_nodes:
        level_map[id(node)] = max_level

        # 计算每层节点数
    level_counts = {}
    for _, level in level_map.items():
        level_counts[level] = level_counts.get(level, 0) + 1

    print(f"树深度: {max_level}, 叶子节点数: {len(leaf_nodes)}")
    print(f"层级分布: {level_counts}")

    # 创建用于可视化的图
    G = nx.DiGraph()

    # 将参数ID转换为十六进制前缀
    def get_hex_prefix(param_id):
        """从参数ID获取前四位十六进制表示"""
        try:
            # 尝试使用哈希函数获取参数ID的哈希值
            import hashlib
            hash_obj = hashlib.md5(param_id.encode('utf-8'))
            return hash_obj.hexdigest()[:4]
        except:
            # 如果失败，简单返回参数ID的前4个字符
            return param_id[:4] if len(param_id) >= 4 else param_id

            # BFS遍历树, 创建图结构

    def build_graph(node, parent_id=None):
        if node is None:
            return

            # 创建唯一节点ID
        node_id = id(node)
        str_node_id = str(node_id)

        # 确定节点标签和属性 - 使用简化的标签
        if node == root:
            label = "Root"  # 简化 Global Root -> Root
            color = "red"
            shape = "o"  # 方形
            size = 3000
            group = "root"
        elif node_id in node_to_model_id:
            # 模型根节点 - 只保留cnnx部分
            model_id = node_to_model_id[node_id]
            # 提取模型名称，不带"Model:"前缀
            if ":" in model_id:
                parts = model_id.split(":")
                label = parts[-1]
            else:
                label = model_id
            color = "orange"
            shape = "o"  # 圆形
            size = 2500
            group = "model"
        elif hasattr(node, 'is_leaf') and node.is_leaf:
            # 叶子节点 - 只保留参数ID的前四位十六进制
            if node_id in node_to_param_id:
                param_id = node_to_param_id[node_id]
                # 获取参数ID的十六进制前缀
                label = get_hex_prefix(param_id)
            else:
                # 没有参数ID时使用节点ID的后四位
                label = str_node_id[-4:]
            color = "lightgreen"
            shape = "o"  # 三角形
            size = 2000
            group = "param"
        else:
            # 内部节点 - 只保留后面的数字
            node_num = str_node_id.split('-')[-1] if '-' in str_node_id else str_node_id[-4:]
            label = node_num
            color = "skyblue"
            shape = "o"  # 圆形
            size = 2000
            group = "internal"

            # 添加节点
        G.add_node(str_node_id,
                   label=label,
                   color=color,
                   level=level_map[node_id],
                   shape=shape,
                   size=size,
                   group=group,
                   is_leaf=hasattr(node, 'is_leaf') and node.is_leaf)

        # 添加边
        if parent_id:
            G.add_edge(parent_id, str_node_id)

            # 处理子节点
        if hasattr(node, 'left') and node.left:
            build_graph(node.left, str_node_id)
        if hasattr(node, 'right') and node.right:
            build_graph(node.right, str_node_id)

            # 构建图

    build_graph(root)

    # 检查图是否为空
    if len(G.nodes()) == 0:
        print("错误：生成的图没有节点，请检查树结构")
        return

    print(f"成功创建图形，包含 {len(G.nodes())} 个节点和 {len(G.edges())} 条边")

    # ======================== 布局算法 ========================

    # 每个节点的子树大小 (以叶子节点数量计算)
    subtree_sizes = {}

    def count_subtree_leaves(node_id):
        """计算以node_id为根的子树的叶子节点数量"""
        children = list(G.successors(node_id))

        # 如果是叶子节点
        if not children:
            if G.nodes[node_id].get('is_leaf', False):
                subtree_sizes[node_id] = 1
                return 1
            else:
                subtree_sizes[node_id] = 0
                return 0

                # 如果是内部节点，计算所有子节点的叶子数总和
        size = sum(count_subtree_leaves(child) for child in children)
        subtree_sizes[node_id] = size
        return size

        # 计算每个节点的子树大小

    for node in G.nodes():
        if not list(G.predecessors(node)):  # 找到根节点
            count_subtree_leaves(node)

            # 水平位置分配

    def assign_x_positions(node_id, start_pos, available_width):
        """分配节点的水平位置，基于子树大小的相对空间分配"""
        children = list(G.successors(node_id))
        node_level = G.nodes[node_id]['level']

        # 设置当前节点的位置
        if node_id not in pos:
            pos[node_id] = (start_pos + available_width / 2, -node_level * 5)

            # 如果没有子节点，返回
        if not children:
            return

            # 分配子节点位置
        total_subtree_size = sum(subtree_sizes[child] for child in children)
        # 最小单位宽度，确保小子树也有足够空间
        min_unit_width = available_width / (len(children) * 2)

        current_pos = start_pos
        for child in children:
            # 计算子树的空间比例，考虑最小宽度
            if total_subtree_size > 0:
                child_width = max(
                    min_unit_width,
                    (subtree_sizes[child] / total_subtree_size) * available_width
                )
            else:
                child_width = available_width / len(children)

                # 递归分配子节点位置
            assign_x_positions(child, current_pos, child_width)
            current_pos += child_width

            # 初始化位置字典

    pos = {}

    # 查找根节点
    root_node_id = None
    for node in G.nodes():
        if not list(G.predecessors(node)):
            root_node_id = node
            break

            # 估算所需的总宽度 - 基于叶子节点数量和每层节点数
    total_leaf_count = len([n for n in G.nodes() if G.nodes[n].get('is_leaf', False)])
    max_nodes_per_level = max(level_counts.values())

    # 使用一个自适应的宽度因子 - 考虑叶子节点数和每层最大节点数
    width_factor = max(total_leaf_count * 6, max_nodes_per_level * 15)

    # 分配位置 - 使用宽度因子确定总宽度
    assign_x_positions(root_node_id, 0, width_factor)

    # ======================== 绘制树 ========================

    # 计算图像尺寸
    max_x = max(x for x, _ in pos.values())
    min_x = min(x for x, _ in pos.values())
    max_y = abs(min(y for _, y in pos.values()))

    # 计算合适的图像尺寸 - 根据节点数量和分布来决定
    fig_width = max(20, (max_x - min_x) / 100 + 5)  # 添加边距
    fig_height = max(10, max_y / 30 + 3)

    plt.figure(figsize=(fig_width, fig_height), dpi=150)
    plt.clf()  # 清除当前图形

    # 绘制边 - 使用弧线减少交叉
    for u, v in G.edges():
        # 计算边的弯曲度 - 基于水平距离
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = abs(x1 - x2)

        # 根据水平距离动态调整弯曲度
        if dx > 20:
            rad = min(0.3, dx / 200)  # 距离越远弯曲越大，但有上限
        else:
            rad = 0.05  # 小距离时使用小弯曲

        # 绘制边
        nx.draw_networkx_edges(G, pos,
                               edgelist=[(u, v)],
                               arrows=True,
                               edge_color="gray",
                               width=1.0,
                               connectionstyle=f'arc3,rad={rad}',
                               arrowstyle="-|>",
                               arrowsize=12,
                               alpha=0.7)

        # 按节点类型分组绘制
    node_groups = {
        'root': [n for n in G.nodes() if G.nodes[n]['group'] == 'root'],
        'model': [n for n in G.nodes() if G.nodes[n]['group'] == 'model'],
        'internal': [n for n in G.nodes() if G.nodes[n]['group'] == 'internal'],
        'param': [n for n in G.nodes() if G.nodes[n]['group'] == 'param']
    }

    # 不同节点组的形状和大小
    shapes = {'root': 'o', 'model': 'o', 'internal': 'o', 'param': 'o'}
    sizes = {'root': 3000, 'model': 2500, 'internal': 2000, 'param': 2000}
    colors = {'root': 'pink', 'model': 'orange', 'internal': 'skyblue', 'param': 'lightgreen'}

    # 绘制各组节点
    for group, nodes in node_groups.items():
        if not nodes:
            continue

        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodes,
                               node_color=[colors[group]] * len(nodes),
                               node_size=sizes[group],
                               node_shape=shapes[group],
                               edgecolors='black',
                               linewidths=1.5 if group == 'root' else 1.0)

        # 不同组节点使用不同字体大小
    font_sizes = {'root': 13, 'model': 12, 'internal': 10, 'param': 9}

    # 绘制标签
    for group, nodes in node_groups.items():
        if not nodes:
            continue

        labels = {n: G.nodes[n]['label'] for n in nodes}

        nx.draw_networkx_labels(G, pos,
                                labels=labels,
                                font_size=font_sizes[group],
                                font_weight='bold',
                                font_color='black',
                                font_family='sans-serif',
                                horizontalalignment='center',
                                verticalalignment='center')

        # 添加图例
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color="pink", label="Global Root"),
        mpatches.Patch(color="orange", label="Model Root"),
        mpatches.Patch(color="skyblue", label="Internal Node"),
        mpatches.Patch(color="lightgreen", label="Parameter Node")
    ]
    plt.legend(handles=legend_elements,
               loc='upper right',
               bbox_to_anchor=(0.85, 0.85),
               fontsize=12)

    # 设置图形参数
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout(pad=0.3)  # 紧凑布局但留有空间

    # 根据文件类型保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 确定文件格式
    file_ext = os.path.splitext(output_file)[1].lower()

    if file_ext == '.svg':
        # SVG格式 - 最佳文本嵌入选项
        plt.savefig(output_file, format='svg', bbox_inches='tight')
        print(f"树形图已保存为SVG格式，文本完全嵌入: {output_file}")
    elif file_ext == '.pdf':
        # PDF格式 - 良好的文本嵌入
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"树形图已保存为PDF格式，文本嵌入: {output_file}")
    else:
        # PNG或其他位图格式 - 使用高DPI
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"树形图已保存为位图格式，使用高DPI (300): {output_file}")

        # 显示图像
    plt.show()

    return G  # 返回图对象以便进一步分析


def save_chameleon_hash_tree(model_tree, filepath):
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


def load_chameleon_hash_tree(filepath):
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

def main():
    # load cht_keys_params
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)

    # load ecdsa keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()

    all_models_data = {}
    model_id_mapping = {}
    # get model id
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/model_id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    print(model_id_mapping)
    # # get encrypted model params
    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        encrypted_model_param = extract_data_from_hash_node(encrypted_path)
        print(encrypted_model_param)
        for name, param in encrypted_model_param.items():
            all_models_data[model_id][name] = param

    # build model verification tree CHT
    model_tree = ChameleonHashTree(cht_keys)
    model_tree.build_from_model_params(all_models_data, ecdsa_private_key)

    return model_tree




if __name__ == "__main__":
    CHT = main()
    #
    # # draw_tree(CHT, output_file="../figure/CHT.png")
    #
    # save_chameleon_hash_tree(CHT, "./CHT.tree")

    # CHT = load_chameleon_hash_tree("./CHT.tree")
    # print(CHT)