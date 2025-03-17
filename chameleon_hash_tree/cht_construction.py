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
                    'rho': current.parent.rho,  # 使用父节点的随机数
                    'delta': current.parent.delta  # 使用父节点的随机数
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

    def audit_leaf(self, leaf_index: int, expected_data: bytes) -> bool:
        """审计叶节点数据是否与预期一致

        Args:
            leaf_index: 叶节点索引
            expected_data: 预期的数据

        Returns:
            数据是否一致
        """
        if leaf_index >= len(self.leaf_nodes):
            return False

        leaf = self.leaf_nodes[leaf_index]
        return leaf.data == expected_data

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
        self.leaf_nodes = []  # 存储叶节点列表

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

        self.leaf_nodes = leaf_nodes  # 保存叶节点列表

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

    def get_merkle_proof(self, leaf_index: int) -> List[Dict]:
        """获取Merkle证明路径

        Args:
            leaf_index: 叶节点索引

        Returns:
            从叶节点到根的证明路径
        """
        # 实现正确的Merkle证明路径
        if leaf_index >= len(self.leaf_nodes):
            raise ValueError("叶节点索引超出范围")

        path = []
        # 这里实现简化版的Merkle证明路径，实际应用需要完整实现
        if len(self.leaf_nodes) > 1:
            sibling_idx = leaf_index + 1 if leaf_index % 2 == 0 else leaf_index - 1
            if 0 <= sibling_idx < len(self.leaf_nodes):
                sibling_hash = self.leaf_nodes[sibling_idx].hash_value
                path.append({
                    'position': 'right' if leaf_index % 2 == 0 else 'left',
                    'hash': sibling_hash
                })

        return path

    # ====================== ECDSA签名实现 (使用ecdsa库) ======================


# def generate_ecdsa_keys() -> Tuple[ecdsa.SigningKey, ecdsa.VerifyingKey]:
#     """生成ECDSA密钥对
#
#     Returns:
#         签名密钥和验证密钥对
#     """
#     # 使用NIST P-256曲线 (secp256r1)
#     sk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
#     vk = sk.verifying_key
#     return sk, vk


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

    # ====================== 客户端验证器 ======================


class ClientVerifier:
    """客户端验证器，用于验证模型参数和CHT路径"""

    def __init__(self, ch_public_keys: PublicKeySet, ecdsa_public_key: ecdsa.VerifyingKey):
        """初始化客户端验证器

        Args:
            ch_public_keys: 变色龙哈希公钥集
            ecdsa_public_key: ECDSA验证密钥
        """
        self.ch_public_keys = ch_public_keys
        self.ecdsa_public_key = ecdsa_public_key
        # 存储已知的原始模型参数
        self.known_model_params = {}

    def register_model_param(self, model_id: int, param_id: int, data: bytes):
        """注册模型参数的预期值，用于后续审计

        Args:
            model_id: 模型ID
            param_id: 参数ID
            data: 参数数据
        """
        if model_id not in self.known_model_params:
            self.known_model_params[model_id] = {}
        self.known_model_params[model_id][param_id] = data

    def verify_cht_path(self, data: bytes, rho: int, delta: int, proof_path: List[Dict],
                        expected_root_hash: bytes) -> Tuple[bool, str]:
        """验证CHT路径

        Args:
            data: 模型参数数据
            rho: 随机数rho
            delta: 随机数delta
            proof_path: CHT证明路径
            expected_root_hash: 预期的根哈希

        Returns:
            (验证是否通过, 错误信息)
        """
        # 计算叶节点哈希
        current_hash = ChameleonHash.hash(data, rho, delta, self.ch_public_keys)
        print(f"  调试: 叶节点哈希计算完成: {current_hash[:4].hex()}")

        # 沿着证明路径计算根哈希
        for i, step in enumerate(proof_path):
            sibling_hash = step['hash']
            sibling_rho = step['rho']
            sibling_delta = step['delta']

            # 按照位置组合哈希
            if step['position'] == 'left':
                combined_data = sibling_hash + current_hash
            else:
                combined_data = current_hash + sibling_hash

                # 计算父节点哈希
            current_hash = ChameleonHash.hash(combined_data, sibling_rho, sibling_delta, self.ch_public_keys)
            print(f"  调试: 内部节点 {i} 哈希计算完成: {current_hash[:4].hex()}")

            # 检查计算得到的根哈希是否与预期的根哈希匹配
        print(f"  调试: 计算得到的根哈希: {current_hash[:4].hex()}")
        print(f"  调试: 预期的根哈希: {expected_root_hash[:4].hex()}")

        if current_hash != expected_root_hash:
            return False, "MODEL_CHT_INVALID: 模型CHT路径验证失败，参数可能被篡改"

        return True, "MODEL_CHT_VALID: 模型CHT路径验证成功"

    def verify_global_signature(self, root_hash: str, timestamp: int, version: int,
                                signature: bytes) -> Tuple[bool, str]:
        """验证全局根哈希签名

        Args:
            root_hash: 全局根哈希
            timestamp: 时间戳
            version: 版本号
            signature: 签名

        Returns:
            (验证是否通过, 错误信息)
        """
        try:
            is_valid = verify_signature(self.ecdsa_public_key, root_hash, timestamp, version, signature)
            if not is_valid:
                return False, "GLOBAL_ROOT_INVALID: 全局根哈希签名验证失败，全局信息可能被篡改"
            return True, "GLOBAL_ROOT_VALID: 全局根哈希签名验证成功"
        except Exception as e:
            return False, f"SIGNATURE_ERROR: 签名验证过程出错 - {str(e)}"

    def verify_merkle_path(self, model_root_hash: bytes, merkle_path: List[Dict],
                           expected_global_root: str) -> Tuple[bool, str]:
        """验证Merkle路径

        Args:
            model_root_hash: 模型CHT根哈希
            merkle_path: Merkle树路径
            expected_global_root: 预期的全局根哈希

        Returns:
            (验证是否通过, 错误信息)
        """
        # 先计算模型根哈希的SHA-256值，作为Merkle树叶节点
        current_hash = hashlib.sha256(model_root_hash).hexdigest()
        print(f"  调试: 叶节点哈希: {current_hash[:8]}")

        # 沿着Merkle路径计算根哈希
        for i, step in enumerate(merkle_path):
            sibling_hash = step['hash']

            # 按照位置组合哈希
            if step['position'] == 'left':
                combined_hash = sibling_hash + current_hash
            else:
                combined_hash = current_hash + sibling_hash

                # 计算父节点哈希
            current_hash = hashlib.sha256(combined_hash.encode()).hexdigest()
            print(f"  调试: Merkle节点 {i} 哈希: {current_hash[:8]}")

            # 检查计算得到的全局根哈希是否与预期的全局根哈希匹配
        print(f"  调试: 计算的全局根哈希: {current_hash[:16]}")
        print(f"  调试: 预期全局根哈希: {expected_global_root[:16]}")

        if current_hash != expected_global_root:
            return False, "GLOBAL_MERKLE_INVALID: 全局Merkle路径验证失败，其他模型可能被篡改"

        return True, "GLOBAL_MERKLE_VALID: 全局Merkle路径验证成功"

    def full_verification(self, data: bytes, rho: int, delta: int, cht_path: List[Dict],
                          merkle_path: List[Dict], expected_model_root: bytes,
                          global_root: str, timestamp: int, version: int,
                          signature: bytes) -> Dict[str, Any]:
        """执行完整验证流程

        Args:
            data: 模型参数数据
            rho, delta: 随机数
            cht_path: CHT证明路径
            merkle_path: Merkle树路径
            expected_model_root: 预期的模型根哈希
            global_root: 全局根哈希
            timestamp: 时间戳
            version: 版本号
            signature: 签名

        Returns:
            验证结果字典
        """
        results = {}

        # 验证全局签名
        global_valid, global_msg = self.verify_global_signature(
            global_root, timestamp, version, signature
        )
        results['global_signature'] = {'valid': global_valid, 'message': global_msg}

        if not global_valid:
            results['overall'] = {'valid': False, 'message': "验证失败: 全局签名无效"}
            return results

            # 验证CHT路径
        print("\n调试: 开始CHT路径验证")
        cht_valid, cht_msg = self.verify_cht_path(
            data, rho, delta, cht_path, expected_model_root
        )
        results['cht_path'] = {'valid': cht_valid, 'message': cht_msg}

        if not cht_valid:
            results['overall'] = {'valid': False, 'message': "验证失败: 模型参数被篡改"}
            return results

            # 验证Merkle路径
        print("\n调试: 开始Merkle路径验证")
        merkle_valid, merkle_msg = self.verify_merkle_path(
            expected_model_root, merkle_path, global_root
        )
        results['merkle_path'] = {'valid': merkle_valid, 'message': merkle_msg}

        if not merkle_valid:
            results['overall'] = {'valid': False, 'message': "验证失败: 其他模型可能被篡改"}
            return results

            # 所有验证都通过
        results['overall'] = {'valid': True, 'message': "验证成功: 所有检查均通过"}
        return results

    def audit_model_params(self, cloud_server) -> Dict[str, List]:
        """审计所有模型参数，检测哪些参数被修改了

        Args:
            cloud_server: 云服务器实例，用于获取当前参数

        Returns:
            被修改的参数列表，按模型分组
        """
        modified_params = {}

        print("\n开始全面审计...")
        # 遍历所有已知的模型参数
        for model_id in self.known_model_params:
            modified_in_model = []

            for param_id in self.known_model_params[model_id]:
                # 获取当前参数
                try:
                    response = cloud_server.get_model_param(model_id, param_id, honest=True)
                    current_param = response['param']
                    expected_param = self.known_model_params[model_id][param_id]

                    # 检查参数是否被修改
                    if current_param != expected_param:
                        modified_in_model.append({
                            'param_id': param_id,
                            'original': expected_param,
                            'current': current_param
                        })
                        print(f"检测到模型 {model_id} 参数 {param_id} 被修改:")
                        print(f"  原始值: {expected_param}")
                        print(f"  当前值: {current_param}")
                except Exception as e:
                    print(f"审计模型 {model_id} 参数 {param_id} 时出错: {str(e)}")

            if modified_in_model:
                modified_params[model_id] = modified_in_model

        if not modified_params:
            print("审计完成: 未发现被修改的参数")
        else:
            print(f"审计完成: 发现 {sum(len(mods) for mods in modified_params.values())} 个参数被修改")

        return modified_params

    # ====================== 模拟云服务器 ======================


class CloudServer:
    """模拟云服务器，可能篡改模型参数"""

    def __init__(self, models_data: List[List[bytes]],
                 model_cht_trees: List[ChameleonHashTree],
                 root_hashes: List[bytes],
                 global_merkle_tree: MerkleTree,
                 global_root_hash: str,
                 timestamp: int,
                 version: int,
                 signature: bytes,
                 ch_keys: PrivateKeySet):
        """初始化云服务器

        Args:
            models_data: 所有模型的参数数据
            model_cht_trees: 所有模型的CHT树
            root_hashes: 所有模型的根哈希
            global_merkle_tree: 全局Merkle树
            global_root_hash: 全局根哈希
            timestamp: 时间戳
            version: 版本号
            signature: 签名
            ch_keys: 变色龙哈希密钥（通常只在服务端持有）
        """
        self.models_data = models_data
        self.model_cht_trees = model_cht_trees
        self.root_hashes = root_hashes
        self.merkle_tree = global_merkle_tree
        self.global_root_hash = global_root_hash
        self.timestamp = timestamp
        self.version = version
        self.signature = signature
        self.ch_keys = ch_keys  # 云服务器通常持有私钥

        # 特殊用途：创建一个未经授权的替代模型
        self.alt_model_data = [f"unauthorized_model_param_{i}".encode() for i in range(4)]

        # 记录被修改的参数
        self.modified_params = {}

    def get_model_param(self, model_id: int, param_id: int,
                        honest: bool = True) -> Dict[str, Any]:
        """获取模型参数及验证所需的信息

        Args:
            model_id: 模型ID
            param_id: 参数ID
            honest: 是否诚实(否则可能篡改数据)

        Returns:
            模型参数及验证信息
        """
        if model_id >= len(self.models_data) or param_id >= len(self.models_data[model_id]):
            raise ValueError("无效的模型ID或参数ID")

            # 获取原始参数
        original_param = self.models_data[model_id][param_id]

        # 获取CHT路径
        cht_path = self.model_cht_trees[model_id].get_proof_path(param_id)

        # 获取叶节点的随机数
        leaf_node = self.model_cht_trees[model_id].leaf_nodes[param_id]
        rho = leaf_node.rho
        delta = leaf_node.delta

        # 获取模型根哈希
        model_root_hash = self.root_hashes[model_id]

        # 获取实际的Merkle路径
        merkle_path = self.merkle_tree.get_merkle_proof(model_id)

        # 如果不诚实，篡改参数
        param = original_param
        if not honest:
            param = f"{original_param.decode()}_tampered".encode()
            print(f"云服务器篡改了模型{model_id}的参数{param_id}:")
            print(f"  原始值: {original_param}")
            print(f"  篡改值: {param}")

            # 返回所有验证需要的信息
        return {
            'param': param,
            'rho': rho,
            'delta': delta,
            'cht_path': cht_path,
            'model_root_hash': model_root_hash,
            'merkle_path': merkle_path,
            'global_root_hash': self.global_root_hash,
            'timestamp': self.timestamp,
            'version': self.version,
            'signature': self.signature
        }

    def get_alt_model_param(self, model_id: int, param_id: int) -> Dict[str, Any]:
        """获取替代模型参数，同时提供原始模型的验证信息（模拟欺骗）

        Args:
            model_id: 原始模型ID（用于获取验证信息）
            param_id: 参数ID

        Returns:
            替代模型参数及原始模型的验证信息
        """
        if model_id >= len(self.models_data) or param_id >= len(self.models_data[model_id]):
            raise ValueError("无效的模型ID或参数ID")

            # 替代模型参数
        alt_param = self.alt_model_data[param_id]

        # 但使用原始模型的验证信息
        response = self.get_model_param(model_id, param_id, honest=True)

        # 替换参数值
        response['param'] = alt_param

        print(f"云服务器使用替代模型而非客户指定的模型:")
        print(f"  客户请求的模型参数: {self.models_data[model_id][param_id]}")
        print(f"  服务器提供的替代参数: {alt_param}")

        return response

    def tamper_signature(self) -> bytes:
        """篡改签名

        Returns:
            篡改后的签名
        """
        # 简单地反转几个字节来篡改签名
        tampered = bytearray(self.signature)
        tampered[0] = (tampered[0] + 1) % 256
        return bytes(tampered)

    def modify_other_model_param(self, target_model_id: int, target_param_id: int) -> bool:
        """修改其他模型参数（非当前使用的模型）

        使用变色龙哈希的可碰撞特性，保持全局哈希值不变

        Args:
            target_model_id: 目标模型ID
            target_param_id: 目标参数ID

        Returns:
            修改是否成功
        """
        if target_model_id >= len(self.models_data) or target_param_id >= len(self.models_data[target_model_id]):
            return False

            # 保存原始参数
        original_param = self.models_data[target_model_id][target_param_id]

        # 创建修改后的参数
        modified_param = f"{original_param.decode()}_secretly_modified".encode()

        # 使用变色龙哈希的可碰撞性质更新目标参数
        success = self.model_cht_trees[target_model_id].update_leaf(target_param_id, modified_param)

        if success:
            # 更新模型数据
            self.models_data[target_model_id][target_param_id] = modified_param

            # 记录修改
            if target_model_id not in self.modified_params:
                self.modified_params[target_model_id] = []
            self.modified_params[target_model_id].append(target_param_id)

            print(f"云服务器悄悄修改了模型 {target_model_id} 的参数 {target_param_id}:")
            print(f"  原始值: {original_param}")
            print(f"  修改后: {modified_param}")
            print(f"  注意: 由于变色龙哈希的可碰撞性，全局哈希值保持不变")

        return success

    # ====================== 主函数：演示 ======================


def main():
    print("=== 基于离散对数的变色龙哈希树安全验证演示 ===\n")

    # 设置随机种子确保结果一致性
    random.seed(42)

    # 加载ECDSA密钥
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()

    print("ECDSA密钥生成完成")
    print(f"ECDSA公钥: {ecdsa_public_key.to_string().hex()[:16]}...")

    # 生成变色龙哈希密钥
    ch_keys = ChameleonHash.key_gen(256)
    print(f"变色龙哈希密钥生成完成 (p = {ch_keys.get_p().bit_length()} 位)")

    # 模拟模型参数数据
    models_data = []
    model_cht_trees = []
    model_root_hashes = []

    # 创建3个示例模型，增加一个模型使演示更丰富
    for model_id in range(3):
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

        # 构建全局Merkle树
    print("\n=== 构建全局Merkle树 ===")
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

    # 创建云服务器实例
    cloud = CloudServer(
        models_data=models_data,
        model_cht_trees=model_cht_trees,
        root_hashes=model_root_hashes,
        global_merkle_tree=merkle_tree,
        global_root_hash=root_hash,
        timestamp=timestamp,
        version=version,
        signature=signature,
        ch_keys=ch_keys
    )

    # 创建客户端验证器
    client = ClientVerifier(
        ch_public_keys=ch_keys.get_public_key_set(),
        ecdsa_public_key=ecdsa_public_key
    )

    # 客户端注册所有原始模型参数，用于后续审计
    print("\n客户端注册所有原始模型参数用于审计...")
    for model_id in range(len(models_data)):
        for param_id in range(len(models_data[model_id])):
            client.register_model_param(
                model_id, param_id, models_data[model_id][param_id]
            )

            # === 演示1: 正常请求参数，验证应该通过 ===
    print("\n\n===== 演示1: 正常请求 =====")
    model_id, param_id = 0, 1
    print(f"客户端请求模型{model_id}的参数{param_id}")

    # 云服务器返回参数和验证信息
    response = cloud.get_model_param(model_id, param_id, honest=True)

    # 客户端验证
    print("\n客户端执行验证:")
    results = client.full_verification(
        data=response['param'],
        rho=response['rho'],
        delta=response['delta'],
        cht_path=response['cht_path'],
        merkle_path=response['merkle_path'],
        expected_model_root=response['model_root_hash'],
        global_root=response['global_root_hash'],
        timestamp=response['timestamp'],
        version=response['version'],
        signature=response['signature']
    )

    # 打印验证结果 - 添加检查避免KeyError
    print(f"  全局签名验证: {'通过' if results['global_signature']['valid'] else '失败'}")
    if 'cht_path' in results:
        print(f"  CHT路径验证: {'通过' if results['cht_path']['valid'] else '失败'}")
    if 'merkle_path' in results:
        print(f"  Merkle路径验证: {'通过' if results['merkle_path']['valid'] else '失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    # === 演示2: 云服务器篡改模型参数 ===
    print("\n\n===== 演示2: 云服务器篡改模型参数 =====")
    model_id, param_id = 1, 2
    print(f"客户端请求模型{model_id}的参数{param_id}")

    # 云服务器返回篡改的参数
    tampered_response = cloud.get_model_param(model_id, param_id, honest=False)

    # 客户端验证
    print("\n客户端执行验证:")
    results = client.full_verification(
        data=tampered_response['param'],
        rho=tampered_response['rho'],
        delta=tampered_response['delta'],
        cht_path=tampered_response['cht_path'],
        merkle_path=tampered_response['merkle_path'],
        expected_model_root=tampered_response['model_root_hash'],
        global_root=tampered_response['global_root_hash'],
        timestamp=tampered_response['timestamp'],
        version=tampered_response['version'],
        signature=tampered_response['signature']
    )

    # 打印验证结果 - 添加检查避免KeyError
    print(f"  全局签名验证: {'通过' if results['global_signature']['valid'] else '失败'}")
    if 'cht_path' in results:
        print(f"  CHT路径验证: {'通过' if results['cht_path']['valid'] else '失败'}")
    if 'merkle_path' in results:
        print(f"  Merkle路径验证: {'通过' if results['merkle_path']['valid'] else '失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")
    print(f"  错误信息: {results.get('overall', {}).get('message', '')}")

    # === 演示3: 云服务器篡改签名 ===
    print("\n\n===== 演示3: 云服务器篡改签名 =====")
    model_id, param_id = 0, 0
    print(f"客户端请求模型{model_id}的参数{param_id}")

    # 云服务器返回参数，但篡改签名
    response = cloud.get_model_param(model_id, param_id, honest=True)
    tampered_signature = cloud.tamper_signature()
    print(f"云服务器篡改了签名:")
    print(f"  原始签名: {response['signature'][:16].hex()}...")
    print(f"  篡改签名: {tampered_signature[:16].hex()}...")

    response['signature'] = tampered_signature

    # 客户端验证
    print("\n客户端执行验证:")
    results = client.full_verification(
        data=response['param'],
        rho=response['rho'],
        delta=response['delta'],
        cht_path=response['cht_path'],
        merkle_path=response['merkle_path'],
        expected_model_root=response['model_root_hash'],
        global_root=response['global_root_hash'],
        timestamp=response['timestamp'],
        version=response['version'],
        signature=response['signature']
    )

    # 打印验证结果 - 添加检查避免KeyError
    print(f"  全局签名验证: {'通过' if results['global_signature']['valid'] else '失败'}")
    if 'cht_path' in results:
        print(f"  CHT路径验证: {'通过' if results['cht_path']['valid'] else '失败'}")
    if 'merkle_path' in results:
        print(f"  Merkle路径验证: {'通过' if results['merkle_path']['valid'] else '失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")
    print(f"  错误信息: {results.get('overall', {}).get('message', '')}")

    # === 演示4: 云服务器使用非指定模型 ===
    print("\n\n===== 演示4: 云服务器使用非指定模型 =====")
    model_id, param_id = 0, 2
    print(f"客户端请求模型{model_id}的参数{param_id}")

    # 云服务器使用替代模型，但提供原始模型的验证信息
    alt_response = cloud.get_alt_model_param(model_id, param_id)

    # 客户端验证
    print("\n客户端执行验证:")
    results = client.full_verification(
        data=alt_response['param'],
        rho=alt_response['rho'],
        delta=alt_response['delta'],
        cht_path=alt_response['cht_path'],
        merkle_path=alt_response['merkle_path'],
        expected_model_root=alt_response['model_root_hash'],
        global_root=alt_response['global_root_hash'],
        timestamp=alt_response['timestamp'],
        version=alt_response['version'],
        signature=alt_response['signature']
    )

    # 打印验证结果
    print(f"  全局签名验证: {'通过' if results['global_signature']['valid'] else '失败'}")
    if 'cht_path' in results:
        print(f"  CHT路径验证: {'通过' if results['cht_path']['valid'] else '失败'}")
    if 'merkle_path' in results:
        print(f"  Merkle路径验证: {'通过' if results['merkle_path']['valid'] else '失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")
    print(f"  错误信息: {results.get('overall', {}).get('message', '')}")

    # === 演示5: 检测非当前请求模型的修改 ===
    print("\n\n===== 演示5: 检测非当前请求模型的修改 =====")

    # 云服务器秘密修改模型2的参数0，客户端将请求模型0的参数
    # 通过变色龙哈希的碰撞特性，修改后整体哈希值不变，常规验证会通过
    target_model_id, target_param_id = 2, 1
    other_model_id, other_param_id = 0, 3

    print(f"云服务器利用变色龙哈希特性悄悄修改模型{target_model_id}的参数{target_param_id}")
    cloud.modify_other_model_param(target_model_id, target_param_id)

    # 客户端请求完全不同的模型参数，验证应该通过
    print(f"\n客户端请求另一个模型{other_model_id}的参数{other_param_id}")
    response = cloud.get_model_param(other_model_id, other_param_id, honest=True)

    # 客户端验证当前请求的参数
    print("\n客户端执行验证:")
    results = client.full_verification(
        data=response['param'],
        rho=response['rho'],
        delta=response['delta'],
        cht_path=response['cht_path'],
        merkle_path=response['merkle_path'],
        expected_model_root=response['model_root_hash'],
        global_root=response['global_root_hash'],
        timestamp=response['timestamp'],
        version=response['version'],
        signature=response['signature']
    )

    # 打印验证结果
    print(f"  全局签名验证: {'通过' if results['global_signature']['valid'] else '失败'}")
    if 'cht_path' in results:
        print(f"  CHT路径验证: {'通过' if results['cht_path']['valid'] else '失败'}")
    if 'merkle_path' in results:
        print(f"  Merkle路径验证: {'通过' if results['merkle_path']['valid'] else '失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    print("\n验证通过，但参数被修改了！客户端现在进行全面审计来检测被修改的参数...")

    # 客户端进行全面审计，检测哪些模型参数被修改
    modified_params = client.audit_model_params(cloud)

    # 打印审计结果
    if modified_params:
        print("\n审计结果:")
        for model_id, params in modified_params.items():
            print(f"  模型 {model_id} 被修改的参数:")
            for param_info in params:
                param_id = param_info['param_id']
                original = param_info['original']
                current = param_info['current']
                print(f"    参数 {param_id}:")
                print(f"      原始值: {original}")
                print(f"      当前值: {current}")
                print(f"      变化: {original} -> {current}")
    else:
        print("审计未发现任何被修改的参数")

        # 客户端可以验证被修改参数的路径是否仍然有效
    if modified_params:
        detected_model_id = list(modified_params.keys())[0]
        detected_param_id = modified_params[detected_model_id][0]['param_id']

        print(f"\n验证被修改的参数 (模型{detected_model_id} 参数{detected_param_id}) 路径是否有效...")

        # 获取被修改参数的信息
        modified_response = cloud.get_model_param(detected_model_id, detected_param_id, honest=True)

        # 使用原始数据进行验证
        original_data = client.known_model_params[detected_model_id][detected_param_id]

        # 检查使用原始数据是否能通过验证（应该失败，因为路径上的随机数已经被修改）
        print("\n使用原始参数验证被修改的路径:")

        try:
            cht_valid, cht_msg = client.verify_cht_path(
                original_data,  # 使用原始数据
                modified_response['rho'],  # 但使用修改后的随机数
                modified_response['delta'],  # 修改后的随机数
                modified_response['cht_path'],  # 修改后的路径
                modified_response['model_root_hash']  # 模型根哈希
            )
            print(f"  验证结果: {'通过' if cht_valid else '失败'}")
            print(f"  消息: {cht_msg}")

            if not cht_valid:
                print("  这证明云服务器使用了变色龙哈希的碰撞特性修改了参数，并调整了随机数")
        except Exception as e:
            print(f"  验证过程出错: {str(e)}")

            # === 演示5的总结 ===
    print("\n===== 演示5总结 =====")
    print("1. 云服务器利用变色龙哈希的碰撞特性修改了模型参数")
    print("2. 修改完成后，全局哈希值保持不变，常规验证仍然通过")
    print("3. 客户端通过全面审计发现了被修改的参数")
    print("4. 云服务器无法同时修改参数和保持随机数不变，这是审计的基础")
    print("5. 尽管变色龙哈希允许碰撞，但结合客户端存储原始参数的审计机制，仍能检测任何修改")

    print("\n=== 安全验证演示完成 ===")


if __name__ == "__main__":
    main()