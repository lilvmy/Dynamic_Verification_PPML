import hashlib
import random
import time
import base64
from typing import List, Dict, Tuple, Any, Optional
import secrets
import ecdsa  # 使用ecdsa专用库
from initialization.setup import load_ecdsa_keys


# ====================== 离散对数变色龙哈希实现 ======================
class PublicKeySet:
    """存储变色龙哈希的公钥和相关参数"""

    def __init__(self, p, q, g, pk):
        self.p = p  # 安全素数
        self.q = q  # 素数q，满足p = 2q + 1
        self.g = g  # 生成器
        self.pk = pk  # 公钥

    def get_p(self):
        return self.p

    def get_q(self):
        return self.q

    def get_g(self):
        return self.g

    def get_public_key(self):
        return self.pk


class PrivateKeySet(PublicKeySet):
    """存储变色龙哈希的私钥和所有相关参数"""

    def __init__(self, p, q, g, sk, pk):
        super().__init__(p, q, g, pk)
        self.sk = sk  # 私钥

    def get_secret_key(self):
        return self.sk

    def get_public_key_set(self):
        return PublicKeySet(self.p, self.q, self.g, self.pk)


class PreImage:
    """存储消息和随机数的前像"""

    def __init__(self, data, rho, delta):
        self.data = data
        self.rho = rho
        self.delta = delta


class ChameleonHash:
    """基于离散对数的变色龙哈希实现"""

    CERTAINTY = 80

    @staticmethod
    def get_safe_prime(t: int) -> int:
        """生成形如2q+1的安全素数

        Args:
            t: 安全参数，素数的位数

        Returns:
            安全素数p
        """
        while True:
            # 生成随机素数q
            q = secrets.randbits(t - 1)  # 确保p的位长为t
            if q % 2 == 0:
                q += 1

                # 根据Miller-Rabin素性测试判断q是否为素数
            if ChameleonHash.is_probable_prime(q, ChameleonHash.CERTAINTY):
                # 计算p = 2q + 1
                p = 2 * q + 1
                # 测试p是否也是素数
                if ChameleonHash.is_probable_prime(p, ChameleonHash.CERTAINTY):
                    return p

    @staticmethod
    def is_probable_prime(n: int, k: int) -> bool:
        """实现Miller-Rabin素性测试

        Args:
            n: 待测试的数
            k: 测试轮数，提高k可以降低误判概率

        Returns:
            n是否可能为素数
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False

            # 将n-1表示为d*2^r
        r, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            r += 1

            # 进行k轮Miller-Rabin测试
        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue

            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False

        return True

    @staticmethod
    def get_random_in_range(n: int) -> int:
        """获取[1, n-1]范围内的随机数

        Args:
            n: 范围上限

        Returns:
            范围内的随机整数
        """
        return random.randint(1, n - 1)

    @staticmethod
    def find_generator(p: int) -> int:
        """寻找模p的生成器

        对于素数p形如2q+1，寻找阶为q的子群的生成器

        Args:
            p: 素数

        Returns:
            模p的生成器
        """
        while True:
            h = ChameleonHash.get_random_in_range(p)
            g = pow(h, 2, p)
            if g != 1:
                return pow(g, 2, p)

    @staticmethod
    def crypto_hash(a: int, b: int, bit_length: int) -> int:
        """可变长度的密码哈希函数

        Args:
            a: 第一个输入整数
            b: 第二个输入整数
            bit_length: 目标哈希位长度

        Returns:
            指定位长的哈希值
        """
        # 连接a和b的字节表示
        x = (a.to_bytes((a.bit_length() + 7) // 8, 'big') +
             b.to_bytes((b.bit_length() + 7) // 8, 'big'))

        # 初始SHA3哈希
        md = hashlib.sha3_256()
        md.update(x)
        hash_value = int.from_bytes(md.digest(), 'big')

        # 如果哈希长度不足，继续哈希并拼接
        while hash_value.bit_length() < bit_length:
            x = (int.from_bytes(x, 'big') + 1).to_bytes((len(x) + 1), 'big')
            md = hashlib.sha3_256()
            md.update(x)
            block = int.from_bytes(md.digest(), 'big')
            hash_value = (hash_value << block.bit_length()) | block

            # 调整到目标位长
        shift_back = hash_value.bit_length() - bit_length
        return hash_value >> shift_back

    @staticmethod
    def key_gen(t: int) -> PrivateKeySet:
        """生成随机公私钥对

        Args:
            t: 安全参数，素数的位数

        Returns:
            包含所有参数的私钥集
        """
        p = ChameleonHash.get_safe_prime(t)
        q = (p - 1) // 2

        g = ChameleonHash.find_generator(p)
        sk = ChameleonHash.get_random_in_range(q)
        pk = pow(g, sk, p)

        return PrivateKeySet(p, q, g, sk, pk)

    @staticmethod
    def hash(data: bytes, rho: int, delta: int, keys: PublicKeySet) -> bytes:
        """计算消息的变色龙哈希值

        Args:
            data: 待哈希的消息
            rho: 随机数
            delta: 随机数
            keys: 公钥集

        Returns:
            哈希值的字节表示
        """
        e = ChameleonHash.crypto_hash(int.from_bytes(data, 'big'), rho, keys.get_p().bit_length())

        t1 = pow(keys.get_public_key(), e, keys.get_p())
        t2 = pow(keys.get_g(), delta, keys.get_p())
        ch = (rho - (t1 * t2) % keys.get_p()) % keys.get_q()

        return ch.to_bytes((ch.bit_length() + 7) // 8, 'big')

    @staticmethod
    def forge(hash_value: bytes, data: bytes, keys: PrivateKeySet) -> PreImage:
        """为新消息伪造前像

        Args:
            hash_value: 原哈希值
            data: 新消息
            keys: 私钥集

        Returns:
            新消息的前像
        """
        c = int.from_bytes(hash_value, 'big')
        m_prime = int.from_bytes(data, 'big')
        k = ChameleonHash.get_random_in_range(keys.get_q())

        rho_prime = (c + pow(keys.get_g(), k, keys.get_p()) % keys.get_q()) % keys.get_q()
        e_prime = ChameleonHash.crypto_hash(m_prime, rho_prime, keys.get_p().bit_length())
        delta_prime = (k - (e_prime * keys.get_secret_key())) % keys.get_q()

        return PreImage(data, rho_prime, delta_prime)

    # ====================== CHT树节点实现 ======================


class CHTNode:
    """变色龙哈希树节点"""

    def __init__(self):
        self.hash_value = None  # 节点哈希值（字节表示）
        self.rho = None  # 随机数rho
        self.delta = None  # 随机数delta
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.parent = None  # 父节点
        self.is_leaf = False  # 是否为叶节点
        self.data = None  # 叶节点数据引用


# ====================== 变色龙哈希树实现 ======================
class ChameleonHashTree:
    """基于离散对数的变色龙哈希树实现"""

    def __init__(self, keys: PrivateKeySet, security_param: int = 512):
        """初始化变色龙哈希树

        Args:
            keys: 变色龙哈希的密钥集
            security_param: 安全参数
        """
        self.keys = keys
        self.security_param = security_param
        self.public_keys = keys.get_public_key_set()
        self.root = None
        self.leaf_nodes = []  # 跟踪所有叶节点

    def build_from_model_params(self, model_params: List[bytes]) -> CHTNode:
        """从模型参数构建CHT树

        Args:
            model_params: 模型参数块列表

        Returns:
            树的根节点
        """
        print(f"构建CHT树，包含 {len(model_params)} 个参数块")

        # 构建叶节点
        leaf_nodes = []
        for i, param_block in enumerate(model_params):
            node = CHTNode()
            node.is_leaf = True
            node.data = param_block

            # 生成随机数
            node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
            node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

            # 计算叶节点哈希值
            node.hash_value = ChameleonHash.hash(param_block, node.rho, node.delta, self.public_keys)

            leaf_nodes.append(node)
            hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])  # 显示前4字节
            print(f"  叶节点 {i}：哈希值 = 0x{hash_str}...")

        self.leaf_nodes = leaf_nodes

        # 构建内部节点层次
        self.root = self._build_internal_nodes(leaf_nodes)
        return self.root

    def _build_internal_nodes(self, nodes: List[CHTNode]) -> CHTNode:
        """递归构建内部节点

        Args:
            nodes: 当前层的节点列表

        Returns:
            当前子树的根节点
        """
        if len(nodes) == 1:
            return nodes[0]

        parent_nodes = []

        # 两两分组创建父节点
        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # 如果存在右节点
            if i + 1 < len(nodes):
                right_node = nodes[i + 1]

                # 创建父节点
                parent = CHTNode()
                parent.left = left_node
                parent.right = right_node
                left_node.parent = parent
                right_node.parent = parent

                # 连接左右子节点的哈希值
                combined_data = left_node.hash_value + right_node.hash_value

                # 生成随机数
                parent.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                parent.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                # 计算父节点哈希
                parent.hash_value = ChameleonHash.hash(combined_data, parent.rho, parent.delta, self.public_keys)

                parent_nodes.append(parent)
                hash_str = ''.join(f'{b:02x}' for b in parent.hash_value[:4])
                print(f"  内部节点：哈希值 = 0x{hash_str}...")
            else:
                # 奇数个节点时，直接提升最后一个节点
                parent_nodes.append(left_node)

                # 递归构建上层节点
        return self._build_internal_nodes(parent_nodes)

    def get_root_hash(self) -> bytes:
        """获取树的根哈希值"""
        if self.root:
            return self.root.hash_value
        return None

    def get_proof_path(self, leaf_index: int) -> List[Dict]:
        """获取指定叶节点到根的证明路径

        Args:
            leaf_index: 叶节点索引

        Returns:
            路径上的节点信息列表
        """
        if leaf_index >= len(self.leaf_nodes):
            raise ValueError("叶节点索引超出范围")

        proof_path = []
        current = self.leaf_nodes[leaf_index]

        while current.parent is not None:
            # 确定当前节点是左节点还是右节点
            is_left = current == current.parent.left

            # 获取兄弟节点
            sibling = current.parent.right if is_left else current.parent.left

            if sibling:
                proof_path.append({
                    'position': 'left' if not is_left else 'right',
                    'hash': sibling.hash_value,
                    'rho': sibling.rho,
                    'delta': sibling.delta
                })

            current = current.parent

        return proof_path

    def update_leaf(self, leaf_index: int, new_data: bytes) -> bool:
        """更新叶节点并重新计算路径上的哈希值

        Args:
            leaf_index: 要更新的叶节点索引
            new_data: 新的模型参数数据

        Returns:
            更新是否成功
        """
        if leaf_index >= len(self.leaf_nodes):
            return False

        leaf = self.leaf_nodes[leaf_index]
        old_data = leaf.data
        old_hash = leaf.hash_value

        # 找到新数据的碰撞
        pre_image = ChameleonHash.forge(old_hash, new_data, self.keys)

        # 更新叶节点
        leaf.data = new_data
        leaf.rho = pre_image.rho
        leaf.delta = pre_image.delta

        # 验证哈希值是否保持不变
        new_hash = ChameleonHash.hash(new_data, pre_image.rho, pre_image.delta, self.public_keys)
        if new_hash != old_hash:
            leaf.data = old_data  # 恢复原始数据
            return False

        hash_str = ''.join(f'{b:02x}' for b in leaf.hash_value[:4])
        print(f"叶节点 {leaf_index} 更新成功，哈希值保持: 0x{hash_str}...")
        return True

    # ====================== Merkle树实现 ======================


class MerkleNode:
    """Merkle树节点"""

    def __init__(self, hash_value=None):
        self.hash_value = hash_value  # 节点哈希值
        self.left = None  # 左子节点
        self.right = None  # 右子节点


class MerkleTree:
    """全局Merkle树实现，用于整合多个模型的CHT根哈希"""

    def __init__(self):
        self.root = None

    def _hash_data(self, data) -> str:
        """计算数据的SHA-256哈希

        Args:
            data: 要哈希的数据

        Returns:
            数据的SHA-256哈希
        """
        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        else:
            return hashlib.sha256(str(data).encode()).hexdigest()

    def build_from_root_hashes(self, root_hashes: List[bytes]) -> MerkleNode:
        """从多个CHT根哈希构建全局Merkle树

        Args:
            root_hashes: CHT根哈希列表

        Returns:
            Merkle树的根节点
        """
        print(f"\n构建全局Merkle树，包含 {len(root_hashes)} 个模型CHT根哈希")

        # 创建叶节点
        leaf_nodes = []
        for i, hash_value in enumerate(root_hashes):
            hash_hex = self._hash_data(hash_value)
            node = MerkleNode(hash_hex)
            leaf_nodes.append(node)
            hash_str = hash_hex[:8]
            print(f"  模型 {i} 根哈希：0x{hash_str}...")

            # 构建内部节点
        self.root = self._build_internal_nodes(leaf_nodes)
        print(f"  全局Merkle树根哈希：{self.root.hash_value[:16]}...")
        return self.root

    def _build_internal_nodes(self, nodes: List[MerkleNode]) -> MerkleNode:
        """递归构建内部节点

        Args:
            nodes: 当前层的节点列表

        Returns:
            当前子树的根节点
        """
        # 基本情况：只有一个节点时返回该节点
        if len(nodes) == 1:
            return nodes[0]

            # 存储上一层父节点
        parent_nodes = []

        # 两两分组创建父节点
        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # 如果存在右节点
            if i + 1 < len(nodes):
                right_node = nodes[i + 1]

                # 创建父节点
                parent = MerkleNode()
                parent.left = left_node
                parent.right = right_node

                # 连接左右子节点的哈希值并计算父节点哈希
                combined_hash = left_node.hash_value + right_node.hash_value
                parent.hash_value = self._hash_data(combined_hash)

                parent_nodes.append(parent)
            else:
                # 奇数个节点时，直接提升最后一个节点
                parent_nodes.append(left_node)

                # 递归构建上层节点
        return self._build_internal_nodes(parent_nodes)

    def get_root_hash(self) -> str:
        """获取全局Merkle树的根哈希值"""
        if self.root:
            return self.root.hash_value
        return None

    # ====================== ECDSA签名实现 (使用ecdsa库) ======================

def sign_root_hash(private_key: ecdsa.SigningKey,
                   root_hash: str,
                   timestamp: int,
                   version: int) -> bytes:
    """对根哈希、时间戳和版本进行签名

    Args:
        private_key: ECDSA签名密钥
        root_hash: Merkle树根哈希
        timestamp: 时间戳
        version: 版本号

    Returns:
        签名
    """
    # 构造要签名的消息
    message = f"{root_hash}|{timestamp}|{version}".encode()

    # 使用SHA-256哈希算法签名
    signature = private_key.sign(message, hashfunc=hashlib.sha256)

    return signature


def verify_signature(public_key: ecdsa.VerifyingKey,
                     root_hash: str,
                     timestamp: int,
                     version: int,
                     signature: bytes) -> bool:
    """验证根哈希签名

    Args:
        public_key: ECDSA验证密钥
        root_hash: Merkle树根哈希
        timestamp: 时间戳
        version: 版本号
        signature: 签名

    Returns:
        验证是否通过
    """
    # 构造要验证的消息
    message = f"{root_hash}|{timestamp}|{version}".encode()

    try:
        # 验证签名
        return public_key.verify(signature, message, hashfunc=hashlib.sha256)
    except ecdsa.BadSignatureError:
        return False

    # ====================== 主函数：演示 ======================


def main():
    print("=== 基于离散对数的变色龙哈希树实现 (使用ecdsa库) ===\n")

    # 生成ECDSA密钥
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    print("ECDSA密钥生成完成")
    print(f"ECDSA公钥: {ecdsa_public_key.to_string().hex()[:16]}...")

    # 步骤2: 为模型生成变色龙哈希密钥并构建CHT
    print("\n=== 步骤2: 为模型构建变色龙哈希树 ===")

    # 生成变色龙哈希密钥（安全参数简化为256位用于演示）
    ch_keys = ChameleonHash.key_gen(256)
    print(f"变色龙哈希密钥生成完成 (p = {ch_keys.get_p().bit_length()} 位)")

    # 模拟模型参数数据
    models_data = []
    model_cht_trees = []
    model_root_hashes = []

    # 创建2个示例模型
    for model_id in range(2):
        print(f"\n构建模型 {model_id} 的CHT:")

        # 模拟模型参数块 (每个模型有4个参数块)
        model_params = [
            f"model_{model_id}_param_{i}".encode()
            for i in range(4)
        ]
        models_data.append(model_params)

        # 构建CHT
        cht = ChameleonHashTree(ch_keys)
        root_node = cht.build_from_model_params(model_params)
        cht.root = root_node

        # 保存树和根哈希
        model_cht_trees.append(cht)
        model_root_hashes.append(cht.get_root_hash())

        hash_str = ''.join(f'{b:02x}' for b in cht.get_root_hash()[:4])
        print(f"模型 {model_id} CHT构建完成，根哈希: 0x{hash_str}...")

        # 步骤3: 构建全局Merkle树
    print("\n=== 步骤3: 构建全局Merkle树 ===")

    # 创建全局Merkle树
    merkle_tree = MerkleTree()
    merkle_tree.build_from_root_hashes(model_root_hashes)
    root_hash = merkle_tree.get_root_hash()

    # 添加版本信息并签名
    timestamp = int(time.time())
    version = 1
    signature = sign_root_hash(ecdsa_private_key, root_hash, timestamp, version)

    print(f"\n全局根哈希签名完成:")
    print(f"  时间戳: {timestamp}")
    print(f"  版本号: {version}")
    print(f"  根哈希: {root_hash[:16]}...")
    print(f"  签名: {signature[:16].hex()}...")

    # 验证签名
    is_valid = verify_signature(ecdsa_public_key, root_hash, timestamp, version, signature)
    print(f"  签名验证: {'成功' if is_valid else '失败'}")

    # 演示更新一个模型参数并使用陷阱门更新
    print("\n=== 演示使用变色龙哈希特性更新模型参数 ===")
    model_id = 0
    param_index = 1
    new_param_value = f"model_{model_id}_param_{param_index}_updated".encode()

    print(f"更新模型 {model_id} 的参数 {param_index}")
    print(f"  原始值: {models_data[model_id][param_index]}")
    print(f"  新值: {new_param_value}")

    # 执行更新
    old_hash = model_cht_trees[model_id].get_root_hash()
    old_hash_str = ''.join(f'{b:02x}' for b in old_hash[:4])
    print(f"  更新前根哈希: 0x{old_hash_str}...")

    success = model_cht_trees[model_id].update_leaf(param_index, new_param_value)

    if success:
        new_hash = model_cht_trees[model_id].get_root_hash()
        new_hash_str = ''.join(f'{b:02x}' for b in new_hash[:4])
        print(f"  更新后根哈希: 0x{new_hash_str}...")

        if old_hash == new_hash:
            print("  验证成功: 使用变色龙哈希特性，模型参数更新后哈希值保持不变!")
        else:
            print("  验证失败: 哈希值发生变化")

            # 重新创建全局Merkle树并签名新版本
        merkle_tree.build_from_root_hashes(model_root_hashes)
        new_root_hash = merkle_tree.get_root_hash()
        new_timestamp = int(time.time())
        new_version = version + 1

        # 使用ecdsa签名
        new_signature = sign_root_hash(ecdsa_private_key, new_root_hash, new_timestamp, new_version)

        print(f"\n参数更新后的全局签名:")
        print(f"  时间戳: {new_timestamp}")
        print(f"  版本号: {new_version}")
        print(f"  根哈希: {new_root_hash[:16]}...")
        print(f"  签名: {new_signature[:16].hex()}...")

        # 验证新签名
        is_valid = verify_signature(ecdsa_public_key, new_root_hash, new_timestamp, new_version, new_signature)
        print(f"  签名验证: {'成功' if is_valid else '失败'}")
    else:
        print("  参数更新失败")

    print("\n=== 步骤2和步骤3完成 ===")


if __name__ == "__main__":
    main()