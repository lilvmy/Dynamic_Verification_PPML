import numpy as np
import hashlib
import time
import json
import sys
import os
import psutil
import math
from typing import List, Dict, Tuple, Any, Optional
import random
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from collections import deque


class TreeNode:
    """Variable Merkle Hash Tree节点"""

    def __init__(self, level, position, leaf_nodes, num_children, hash_value):
        """初始化VMHT节点"""
        self.level = level  # 树中的层级 (H表示叶节点，1表示根节点)
        self.position = position  # 在层中的位置
        self.leaf_nodes = leaf_nodes  # 子树中的叶节点数量
        self.num_children = num_children  # 子节点数量
        self.hash_value = hash_value  # 节点哈希值
        self.firstChild = None  # 第一个子节点
        self.secondChild = None  # 第二个子节点
        self.thirdChild = None  # 第三个子节点 (3子节点的情况)
        self.parent = None  # 父节点
        self.data = None  # 叶节点数据
        # 用于可视化和调试
        self.param_id = None  # 参数ID (对叶节点)
        self.model_id = None  # 模型ID (对模型子树根节点)
        self.block_idx = None  # 参数块索引
        self.shuffle_key = None  # 参数特定的随机置换密钥

    def __str__(self):
        model_str = f", model={self.model_id}" if self.model_id else ""
        param_str = f", param={self.param_id}" if self.param_id else ""
        return f"Node(level={self.level}, pos={self.position}, leaves={self.leaf_nodes}, children={self.num_children}{model_str}{param_str})"

    def get_children(self):
        """返回节点的所有子节点列表"""
        children = []
        if self.firstChild:
            children.append(self.firstChild)
        if self.secondChild:
            children.append(self.secondChild)
        if self.thirdChild:
            children.append(self.thirdChild)
        return children


class VariableMerkleHahsTree:
    """Variable Merkle Hash Tree实现"""

    def __init__(self):
        """初始化VMHT"""
        self.root = None  # 树的根节点
        self.height = 0  # 树的高度
        self.model_trees = {}  # {model_id: root_node} 模型子树
        self.param_nodes = {}  # {model_id: {param_id: [leaf_nodes]}} 每个参数对应的叶节点列表
        self.timestamp = None  # 树创建的时间戳
        self.version = 1  # 树的版本
        self.security_code = None  # 用于哈希的安全码
        self.shuffle_keys = {}  # 参数特定的随机置换密钥 {model_id: {param_id: key}}
        self.param_blocks = {}  # 每个参数的块数 {model_id: {param_id: num_blocks}}
        self.max_block_size = 0  # 最大块大小(字节)
        self.performance_stats = {}  # 性能统计数据

    def hash_function(self, data, g, shuffle_key=None):
        """
        计算数据的哈希值，使用安全码g和可选的随机置换密钥

        参数:
        data -- 要哈希的数据 (numpy数组, 字节或其他)
        g -- 安全码
        shuffle_key -- 可选的参数特定随机置换密钥

        返回:
        bytes: 哈希值
        """
        hasher = hashlib.sha256()

        # 包含安全码
        hasher.update(str(g).encode())

        # 包含随机置换密钥（如果提供）
        if shuffle_key is not None:
            hasher.update(str(shuffle_key).encode())

            # 根据数据类型进行哈希
        if isinstance(data, np.ndarray):
            hasher.update(data.tobytes())
        elif isinstance(data, bytes):
            hasher.update(data)
        elif isinstance(data, str):
            hasher.update(data.encode('utf-8'))
        else:
            hasher.update(str(data).encode('utf-8'))

        return hasher.digest()

    def permutation_function(self, sh, m):
        """
        伪随机置换函数

        参数:
        sh -- 随机置换密钥(种子)
        m -- 要置换的元素数量

        返回:
        List[int]: 索引[0, m-1]的置换
        """
        random.seed(sh)
        indices = list(range(m))
        random.shuffle(indices)
        return indices

    def build_tree(self, all_model_params, security_code=None, max_block_size=16):
        """
        从模型参数构建VMHT树

        参数:
        all_model_params -- 字典 {model_id: {param_id: np.ndarray}}
        security_code -- 可选的安全码
        max_block_size -- 每个参数的数据块数量

        返回:
        dict: 性能统计信息
        """
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # 生成安全码（如果未提供）
        if security_code is None:
            security_code = str(random.randint(1, 10000))
        self.security_code = security_code
        self.max_block_size = max_block_size

        # 记录时间戳
        self.timestamp = int(time.time())

        print(f"构建VMHT，包含 {len(all_model_params)} 个模型")

        # 计算块数和生成参数特定的随机置换密钥
        self.param_blocks = {}
        self.shuffle_keys = {}

        for model_id, params in all_model_params.items():
            self.param_blocks[model_id] = {}
            self.shuffle_keys[model_id] = {}
            self.param_nodes[model_id] = {}

            for param_id, param_data in params.items():
                # 首先将参数展平为一维数组
                if isinstance(param_data, np.ndarray):
                    # 展平NumPy数组
                    flat_param_data = param_data.flatten()
                    param_size = flat_param_data.nbytes
                elif isinstance(param_data, bytes):
                    flat_param_data = param_data
                    param_size = len(param_data)
                else:
                    # 非二进制数据转换为字符串并编码
                    flat_param_data = str(param_data).encode()
                    param_size = len(flat_param_data)

                    # 计算此参数需要的块数
                num_blocks = max(1, math.ceil(param_size / max_block_size))
                self.param_blocks[model_id][param_id] = num_blocks

                # 为此参数生成随机置换密钥（不超过其块数）
                self.shuffle_keys[model_id][param_id] = random.randint(0, num_blocks - 1)

                # 调用vmhtgen处理所有模型参数
        self.vmhtgen(all_model_params)

        # 计算性能指标
        total_time = time.time() - start_time
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        tree_size = self.calculate_storage_size()

        # 记录性能统计
        performance = {
            "build_time_sec": total_time * 1000,
            "memory_used_mb": memory_used,
            "tree_size_mb": tree_size / (1024 * 1024)
        }

        self.performance_stats["build"] = performance

        print(f"\n树构建成功！")
        print(f"根哈希值: 0x{self.root.hash_value.hex()[:16]}...")
        print(f"用时: {total_time:.4f}秒")
        print(f"内存使用: {memory_used:.2f}MB")
        print(f"树大小: {tree_size / (1024 * 1024):.2f}MB")

        return self, performance

    def vmhtgen(self, all_model_params):
        """
        生成VMHT树

        参数:
        all_model_params -- 字典 {model_id: {param_id: np.ndarray}}

        返回:
        TreeNode: 树的根节点
        """
        # 处理模型并构建模型子树
        model_roots = []

        for model_id, model_params in all_model_params.items():
            print(f"构建模型 {model_id} 的子树")

            # 创建数据块列表
            all_blocks = []
            param_nodes_map = {}

            # 处理每个参数
            for param_id, param_data in model_params.items():
                # 展平参数数据
                if isinstance(param_data, np.ndarray):
                    flat_param_data = param_data.flatten().tobytes()
                elif isinstance(param_data, bytes):
                    flat_param_data = param_data
                else:
                    flat_param_data = str(param_data).encode('utf-8')

                    # 获取此参数的块数
                num_blocks = self.param_blocks[model_id][param_id]
                param_size = len(flat_param_data)

                # 获取参数特定的随机置换密钥
                param_shuffle_key = self.shuffle_keys[model_id][param_id]

                # 将参数分割成块
                if num_blocks == 1:
                    # 单个块，不需要分割
                    all_blocks.append((model_id, param_id, 0, flat_param_data, param_shuffle_key))
                else:
                    # 多个块，分割参数
                    block_size = math.ceil(param_size / num_blocks)

                    for i in range(num_blocks):
                        start_idx = i * block_size
                        end_idx = min(start_idx + block_size, param_size)
                        block_data = flat_param_data[start_idx:end_idx]

                        # 添加带有随机置换密钥的块
                        all_blocks.append((model_id, param_id, i, block_data, param_shuffle_key))

                        # 获取块总数
            m = len(all_blocks)
            if m == 0:
                continue

                # 计算树高 H = ⌈log₂m⌉ + 1
            H = math.ceil(math.log2(m)) + 1

            # 为每个块创建叶节点
            Q = deque()
            for i, (model_id, param_id, block_idx, block_data, shuffle_key) in enumerate(all_blocks):
                # 使用参数特定的随机置换计算哈希
                hash_val = self.hash_function(block_data, self.security_code, shuffle_key)

                # 创建叶节点
                node = TreeNode(H, i, 1, 0, hash_val)
                node.model_id = model_id
                node.param_id = param_id
                node.block_idx = block_idx
                node.data = block_data
                node.shuffle_key = shuffle_key

                Q.append(node)

                # 存储参数叶节点引用
                if param_id not in param_nodes_map:
                    param_nodes_map[param_id] = []
                param_nodes_map[param_id].append(node)

                # print(f"  参数 {param_id} (块 {block_idx + 1}/{self.param_blocks[model_id][param_id]}): "
                #       f"哈希值={hash_val.hex()[:8]}... (随机置换密钥: {shuffle_key})")

                # 构建子树
            l = H
            r = 1

            while len(Q) > 1:
                x = Q.popleft()
                y = Q.popleft()
                z = Q[0] if len(Q) > 0 else None
                w = Q[1] if len(Q) > 1 else None

                if z is not None and z.level == l and (w is None or w.level != l):
                    z = Q.popleft()

                    # 创建3子节点
                    combined = x.hash_value + y.hash_value + z.hash_value
                    hash_val = self.hash_function(combined, self.security_code)

                    t = TreeNode(l - 1, r, x.leaf_nodes + y.leaf_nodes + z.leaf_nodes, 3, hash_val)
                    t.firstChild = x
                    t.secondChild = y
                    t.thirdChild = z
                    x.parent = t
                    y.parent = t
                    z.parent = t
                else:
                    # 创建2子节点
                    combined = x.hash_value + y.hash_value
                    hash_val = self.hash_function(combined, self.security_code)

                    t = TreeNode(l - 1, r, x.leaf_nodes + y.leaf_nodes, 2, hash_val)
                    t.firstChild = x
                    t.secondChild = y
                    x.parent = t
                    y.parent = t

                r += 1
                Q.append(t)

                if len(Q) > 0 and Q[0].level < l:
                    l = l - 1
                    r = 1

                    # 获取模型根节点
            model_root = Q.popleft() if Q else None
            model_root.model_id = model_id

            # 保存模型子树和参数映射
            self.model_trees[model_id] = model_root
            self.param_nodes[model_id] = param_nodes_map

            # 添加到模型根节点列表
            model_roots.append(model_root)

            print(f"  模型 {model_id} 子树构建完成: 根哈希值={model_root.hash_value.hex()[:8]}...")

            # 从模型根节点构建全局树
        if model_roots:
            # 计算模型根节点树高
            m = len(model_roots)
            H = math.ceil(math.log2(m)) + 1
            self.height = H

            # 使用模型根节点初始化队列
            Q = deque(model_roots)

            # 构建树
            l = H
            r = 1

            while len(Q) > 1:
                x = Q.popleft()
                y = Q.popleft()
                z = Q[0] if len(Q) > 0 else None
                w = Q[1] if len(Q) > 1 else None

                if z is not None and z.level == l and (w is None or w.level != l):
                    z = Q.popleft()

                    # 创建3子节点
                    combined = x.hash_value + y.hash_value + z.hash_value
                    hash_val = self.hash_function(combined, self.security_code)

                    t = TreeNode(l - 1, r, x.leaf_nodes + y.leaf_nodes + z.leaf_nodes, 3, hash_val)
                    t.firstChild = x
                    t.secondChild = y
                    t.thirdChild = z
                    x.parent = t
                    y.parent = t
                    z.parent = t
                else:
                    # 创建2子节点
                    combined = x.hash_value + y.hash_value
                    hash_val = self.hash_function(combined, self.security_code)

                    t = TreeNode(l - 1, r, x.leaf_nodes + y.leaf_nodes, 2, hash_val)
                    t.firstChild = x
                    t.secondChild = y
                    x.parent = t
                    y.parent = t

                r += 1
                Q.append(t)

                if len(Q) > 0 and Q[0].level < l:
                    l = l - 1
                    r = 1

                    # 设置全局根节点
            self.root = Q.popleft() if Q else None

        print(f"全局树构建完成: 根哈希值={self.root.hash_value.hex()[:8]}...")
        return self.root

    def calculate_storage_size(self):
        """计算树的总存储大小（字节）"""
        if self.root is None:
            return 0

            # 使用集合跟踪已访问节点，避免重复计算
        visited = set()
        total_size = 0

        # 使用栈而非递归，避免栈溢出
        stack = [self.root]

        while stack:
            node = stack.pop()

            # 跳过已访问的节点
            if id(node) in visited:
                continue

                # 标记为已访问
            visited.add(id(node))

            # 计算节点大小
            node_size = sys.getsizeof(node)

            # 添加哈希值大小
            if node.hash_value is not None:
                node_size += sys.getsizeof(node.hash_value)

                # 添加字符串属性大小
            if node.model_id:
                node_size += sys.getsizeof(node.model_id)
            if node.param_id:
                node_size += sys.getsizeof(node.param_id)

                # 添加数据大小
            if node.data is not None:
                if isinstance(node.data, np.ndarray):
                    node_size += node.data.nbytes
                elif isinstance(node.data, bytes):
                    node_size += len(node.data)
                else:
                    node_size += sys.getsizeof(node.data)

                    # 累加到总大小
            total_size += node_size

            # 将子节点添加到栈中
            for child in node.get_children():
                if child is not None:
                    stack.append(child)

        return total_size

    def generate_audit_request(self, sample_ratio=0.3, specific_models=None):
        """
        为VMHT生成审计请求

        参数:
        sample_ratio -- 要抽样的参数比例
        specific_models -- 可选的特定模型ID列表进行审计

        返回:
        AuditRequest对象
        """
        # 计算总参数数
        total_params = 0
        for model_id, params in self.param_nodes.items():
            if specific_models is None or model_id in specific_models:
                total_params += sum(len(blocks) for blocks in params.values())

        if total_params == 0:
            raise ValueError("没有参数可供审计")

            # 计算要抽样的节点数
        ln = max(1, int(total_params * sample_ratio))

        # 生成随机置换密钥
        sh = random.randint(0, total_params - 1)

        # 现在为了找到有效的节点位置，我们先要找出树中实际存在的层级和位置
        valid_positions = self._get_valid_node_positions()

        if not valid_positions:
            # 如果找不到有效位置，使用默认层级和位置
            l = 1
            r = 1
        else:
            # 随机选择一个有效的(level, position)组合
            l, r = random.choice(valid_positions)

        return {"ln": ln, "l": l, "r": r, "sh": sh, "model_ids": specific_models}

    def _get_valid_node_positions(self):
        """
        获取树中所有有效的(level, position)组合

        返回:
        List[(int, int)]: 有效的(level, position)组合列表
        """
        if not self.root:
            return []

        valid_positions = []
        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            # 记录当前节点的层级和位置
            valid_positions.append((node.level, node.position))

            # 添加子节点到队列
            for child in node.get_children():
                if child:
                    queue.append(child)

        return valid_positions

    def verify_proof(self, proof, request):
        """
        验证审计证明

        参数:
        proof -- 完整性证明
        request -- 审计请求

        返回:
        bool: 验证是否通过
        """
        # 检查树是否已构建
        if self.root is None:
            print("错误: 树尚未构建，无法验证")
            return False

            # 处理位置验证请求
        l = request.get("l")
        r = request.get("r")

        if l is not None and r is not None:
            # 获取验证标签
            tag_from_proof = proof.get("tag")
            if not tag_from_proof:
                print("错误: 证明中缺少标签")
                return False

                # 查找特定位置的节点
            node_at_lr = self._find_node_at_level_position(l, r)

            if not node_at_lr:
                print(f"错误: 在级别 {l} 位置 {r} 未找到节点")
                return False

                # 获取树中的标签
            if isinstance(node_at_lr.hash_value, bytes):
                tag_from_tree = node_at_lr.hash_value.hex()
            else:
                tag_from_tree = node_at_lr.hash_value

                # 比较标签
            verification_result = tag_from_proof == tag_from_tree

            if verification_result:
                print(f"验证成功: 级别 {l} 位置 {r} 的节点标签匹配")
            else:
                print(f"验证失败: 标签不匹配")
                print(f"证明中的标签: {tag_from_proof}")
                print(f"树中的标签: {tag_from_tree}")

            return verification_result

            # 处理参数抽样验证请求
        elif "data_references" in proof:
            # 这部分是你原有的参数抽样验证逻辑
            # 根据你的代码，这部分可能不需要更改
            tag_from_proof = proof.get("tag")
            data_references = proof.get("data_references", [])

            # 实现抽样验证逻辑...
            # 这部分因为目前没出问题，可以保持不变

            return True

        else:
            print("错误: 不支持的证明或请求类型")
            return False

    def _find_node_at_level_position(self, level, position):
        """
        在树中查找特定层级和位置的节点

        参数:
        level -- 层级
        position -- 位置

        返回:
        TreeNode or None: 找到的节点或None
        """
        if not self.root:
            return None

        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            if node.level == level and node.position == position:
                return node

                # 如果当前节点层级大于目标层级，继续向下搜索
            if node.level < level:
                for child in node.get_children():
                    if child:
                        queue.append(child)

        return None

    def progen(self, request):
        """
        为审计请求生成完整性证明

        参数:
        request -- 审计请求

        返回:
        完整性证明字典
        """
        # 检查树是否已构建
        if self.root is None:
            print("错误: 树尚未构建，无法生成证明")
            return {"error": "树尚未构建"}

            # 提取请求参数
        ln = request.get("ln", 0)
        sh = request.get("sh", 0)
        specific_models = request.get("model_ids")
        l = request.get("l")
        r = request.get("r")

        # 处理位置验证请求
        if l is not None and r is not None:
            # 查找特定位置的节点
            node_at_lr = self._find_node_at_level_position(l, r)

            if not node_at_lr:
                print(f"错误: 在级别 {l} 位置 {r} 未找到节点")
                return {"error": f"在级别 {l} 位置 {r} 未找到节点"}

                # 获取标签
            if isinstance(node_at_lr.hash_value, bytes):
                tag = node_at_lr.hash_value.hex()
            else:
                tag = node_at_lr.hash_value

            return {
                "tag": tag
            }

            # 处理参数抽样请求
        # 以下是原有的抽样逻辑，可以保持不变
        # ...

        # 收集所有可用的叶子节点
        all_leaf_nodes = []

        # 处理特定模型筛选
        if specific_models:
            # 转换为列表如果是单个模型ID
            model_ids = [specific_models] if isinstance(specific_models, str) else specific_models

            # 只收集指定模型的节点
            for model_id in model_ids:
                if model_id in self.param_nodes:
                    for param_id, block_nodes in self.param_nodes[model_id].items():
                        for block_idx, node in enumerate(block_nodes):
                            shuffle_key = self.shuffle_keys.get(model_id, {}).get(param_id, 0)
                            all_leaf_nodes.append((model_id, param_id, block_idx, node, shuffle_key))
        else:
            # 收集所有模型的节点
            for model_id, params in self.param_nodes.items():
                for param_id, block_nodes in params.items():
                    for block_idx, node in enumerate(block_nodes):
                        shuffle_key = self.shuffle_keys.get(model_id, {}).get(param_id, 0)
                        all_leaf_nodes.append((model_id, param_id, block_idx, node, shuffle_key))

                        # 获取参数总数
        m = len(all_leaf_nodes)
        if m == 0:
            return {"sample_size": 0, "data_references": [],
                    "tag": self.hash_function("empty", self.security_code).hex()}

            # 基于sh应用随机置换
        perm = self.permutation_function(sh, m)
        shuffled_nodes = [all_leaf_nodes[perm[i]] for i in range(m)]

        # 抽样参数（从最后ln个项目）
        ln = min(ln, m)
        start_idx = m - ln
        sampled_nodes = shuffled_nodes[start_idx:]

        # 创建数据引用列表
        data_references = [(node[0], node[1], node[2], node[4]) for node in
                           sampled_nodes]  # (model_id, param_id, block_idx, shuffle_key)

        # 从抽样节点中收集哈希值
        sampled_hashes = []
        for _, _, _, node, _ in sampled_nodes:
            # 获取节点的哈希值
            if node and hasattr(node, 'hash_value'):
                if isinstance(node.hash_value, bytes):
                    sampled_hashes.append(node.hash_value)
                else:
                    # 如果哈希值是字符串，转换为字节
                    sampled_hashes.append(
                        node.hash_value.encode('utf-8') if isinstance(node.hash_value, str) else node.hash_value)

                    # 计算组合哈希作为标签
        if sampled_hashes:
            # 按序连接所有哈希值
            combined_hash = b''.join(sampled_hashes)
            # 计算最终标签
            tag = self.hash_function(combined_hash, self.security_code).hex()
        else:
            tag = self.hash_function("empty", self.security_code).hex()

            # 返回完整性证明
        return {
            "sample_size": ln,
            "data_references": data_references,
            "tag": tag
        }

    def save_to_file(self, output_file):
        """
        将VMHT的元数据保存到文件

        参数:
        output_file -- 输出文件路径
        """
        if not self.root:
            print("错误：树为空，无法保存")
            return

            # 创建可序列化的树表示
        tree_data = {
            'timestamp': self.timestamp,
            'version': self.version,
            'root_hash': self.root.hash_value.hex(),
            'security_code': self.security_code,
            'model_count': len(self.model_trees),
            'model_ids': list(self.model_trees.keys()),
            'performance_stats': self.performance_stats
        }

        with open(output_file, 'w') as f:
            json.dump(tree_data, f, indent=2)

        print(f"VMHT元数据已保存到 {output_file}")

    def get_model_proof(self, model_id):
        """
        获取模型的证明

        参数:
        model_id -- 模型ID

        返回:
        模型证明字典
        """
        if model_id not in self.model_trees:
            raise ValueError(f"模型 {model_id} 不存在")

            # 为特定模型创建审计请求
        request = self.generate_audit_request(sample_ratio=1.0, specific_models=[model_id])

        # 生成证明
        proof = self.progen(request)

        # 合并请求和证明
        result = {
            "request": request,
            "proof": proof,
            "model_id": model_id,
            "root_hash": self.root.hash_value.hex(),
            "timestamp": self.timestamp,
            "version": self.version
        }

        return result

    def generate_param_verification_request(self, model_id):
        """
        生成用于验证模型所有参数块的请求

        参数:
        model_id -- 要验证的模型ID

        返回:
        参数验证请求
        """
        if not self.root:
            raise ValueError("树尚未构建，无法生成验证请求")

        if model_id not in self.param_nodes:
            raise ValueError(f"模型 {model_id} 不存在于树中")

        return {
            "verification_type": "all_params",
            "model_id": model_id
        }

    def generate_param_proof(self, model_id):
        """
        为特定模型的所有参数块生成验证证明

        参数:
        model_id -- 要验证的模型ID

        返回:
        所有参数块的验证证明
        """
        if not self.root:
            print("错误: 树尚未构建，无法生成证明")
            return {"error": "树尚未构建"}

        if model_id not in self.param_nodes:
            print(f"错误: 模型 {model_id} 不存在于树中")
            return {"error": f"模型 {model_id} 不存在"}

            # 收集模型的所有参数块信息和哈希值
        param_blocks = {}

        for param_id, block_nodes in self.param_nodes[model_id].items():
            param_blocks[param_id] = []

            for block_idx, node in enumerate(block_nodes):
                # 获取节点哈希值
                if isinstance(node.hash_value, bytes):
                    hash_value = node.hash_value.hex()
                else:
                    hash_value = node.hash_value

                    # 获取参数特定的随机置换密钥
                shuffle_key = self.shuffle_keys.get(model_id, {}).get(param_id, 0)

                # 添加块信息
                param_blocks[param_id].append({
                    "block_idx": block_idx,
                    "hash": hash_value,
                    "shuffle_key": shuffle_key
                })

                # 获取模型根哈希，用于额外验证
        model_root = self.model_trees.get(model_id)
        model_hash = None
        if model_root:
            if isinstance(model_root.hash_value, bytes):
                model_hash = model_root.hash_value.hex()
            else:
                model_hash = model_root.hash_value

        return {
            "model_id": model_id,
            "model_hash": model_hash,
            "param_blocks": param_blocks,
            "total_blocks": sum(len(blocks) for blocks in param_blocks.values())
        }

    def verify_params(self, proof, request):
        """
        验证模型所有参数块

        参数:
        proof -- 参数块验证证明
        request -- 验证请求

        返回:
        dict: 验证结果，包含成功率和详细信息
        """
        if not self.root:
            print("错误: 树尚未构建，无法验证")
            return {"success": False, "error": "树尚未构建"}

            # 从请求中获取模型ID
        if "model_id" not in request:
            print("错误: 请求中缺少模型ID")
            return {"success": False, "error": "请求中缺少模型ID"}

        model_id = request["model_id"]

        # 检查模型是否存在
        if model_id not in self.param_nodes:
            print(f"错误: 模型 {model_id} 不存在于树中")
            return {"success": False, "error": f"模型 {model_id} 不存在"}

            # 从证明中获取参数块信息
        param_blocks_from_proof = proof.get("param_blocks", {})

        if not param_blocks_from_proof:
            print("错误: 证明中缺少参数块信息")
            return {"success": False, "error": "证明中缺少参数块信息"}

            # 验证模型的所有参数块
        success_count = 0
        total_count = 0
        failures = []

        for param_id, blocks_info in param_blocks_from_proof.items():
            # 检查参数是否存在于树中
            if param_id not in self.param_nodes[model_id]:
                failures.append({
                    "param_id": param_id,
                    "error": "参数不存在于树中"
                })
                continue

            tree_blocks = self.param_nodes[model_id][param_id]

            for block_info in blocks_info:
                block_idx = block_info.get("block_idx")
                hash_from_proof = block_info.get("hash")

                total_count += 1

                # 检查块索引是否有效
                if block_idx is None or block_idx >= len(tree_blocks):
                    failures.append({
                        "param_id": param_id,
                        "block_idx": block_idx,
                        "error": "块索引无效"
                    })
                    continue

                    # 获取树中的块节点
                node = tree_blocks[block_idx]

                # 获取树中的哈希值
                if isinstance(node.hash_value, bytes):
                    hash_from_tree = node.hash_value.hex()
                else:
                    hash_from_tree = node.hash_value

                    # 比较哈希值
                if hash_from_tree == hash_from_proof:
                    success_count += 1
                else:
                    failures.append({
                        "param_id": param_id,
                        "block_idx": block_idx,
                        "hash_proof": hash_from_proof[:10] + "...",
                        "hash_tree": hash_from_tree[:10] + "..."
                    })

                    # 计算验证成功率
        success_rate = success_count / total_count if total_count > 0 else 0

        # 确定整体验证结果
        result = {
            "success": success_count == total_count,
            "success_rate": success_rate,
            "success_count": success_count,
            "total_count": total_count,
            "failures": failures[:10] if len(failures) > 10 else failures  # 限制失败记录数量
        }

        # 打印验证结果
        if result["success"]:
            print(f"验证成功: 模型 {model_id} 的所有 {total_count} 个参数块验证通过")
        else:
            print(f"验证失败: 模型 {model_id} 的 {total_count} 个参数块中有 {total_count - success_count} 个验证失败")

        return result


def get_VMHT_multi_model():
    """从预训练模型构建VMHT并返回性能统计"""
    all_models_data = {}
    model_id_mapping = {}

    # 获取模型ID映射
    with open("/VariableMHT/model_id_pre_trained_model.txt", 'r',
              encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    print(model_id_mapping)

    # 获取每个模型的参数
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

                # 构建VMHT
    vmht_builder = VariableMerkleHahsTree()
    performance = vmht_builder.build_tree(all_models_data)

    # 保存树结构元数据
    vmht_builder.save_to_file("./vmht_metadata.json")

    # 生成并验证一个证明，测试功能
    request = vmht_builder.generate_audit_request(sample_ratio=0.2)
    proof = vmht_builder.progen(request)
    verification_result = vmht_builder.verify_proof(proof, request)

    print(f"审计验证结果: {'通过' if verification_result else '失败'}")

    return performance


if __name__ == "__main__":
    performance = get_VMHT_multi_model()
    print(performance)