import hashlib
import random
import time
import base64
from typing import List, Dict, Tuple, Any, Optional
import secrets
import ecdsa  # 依赖ecdsa库


# ====================== 密钥生成与管理 ======================

def load_ecdsa_keys():
    """生成ECDSA密钥对用于测试"""
    private_key = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
    public_key = private_key.get_verifying_key()
    return private_key, public_key


class PublicKeySet:
    """变色龙哈希公钥集"""

    def __init__(self, p, q, g, pk):
        self.p = p  # 安全素数
        self.q = q  # 素数q, p = 2q + 1
        self.g = g  # 生成元
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
    """变色龙哈希私钥集及相关参数"""

    def __init__(self, p, q, g, sk, pk):
        super().__init__(p, q, g, pk)
        self.sk = sk  # 私钥

    def get_secret_key(self):
        return self.sk

    def get_public_key_set(self):
        return PublicKeySet(self.p, self.q, self.g, self.pk)


class PreImage:
    """存储消息和随机数原像"""

    def __init__(self, data, rho, delta):
        self.data = data
        self.rho = rho
        self.delta = delta

    # ====================== 变色龙哈希实现 ======================


class ChameleonHash:
    """基于离散对数的变色龙哈希实现"""

    CERTAINTY = 80  # 素数测试的确定性

    @staticmethod
    def get_safe_prime(t: int) -> int:
        """生成安全素数p, p=2q+1"""
        while True:
            # 生成随机素数q
            q = secrets.randbits(t - 1)  # q的字节长度为t
            if q % 2 == 0:
                q += 1

                # 基于Miller-Rabin测试判断q是否为素数
            if ChameleonHash.is_probable_prime(q, ChameleonHash.CERTAINTY):
                p = 2 * q + 1
                # 测试p是否为素数
                if ChameleonHash.is_probable_prime(p, ChameleonHash.CERTAINTY):
                    return p

    @staticmethod
    def is_probable_prime(n: int, k: int) -> bool:
        """实现Miller-Rabin素数测试"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False

            # 定义n - 1 = d*2^r
        r, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            r += 1

            # k轮Miller-Rabin测试
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
        """从[1, n-1]范围内获取随机数"""
        return random.randint(1, n - 1)

    @staticmethod
    def find_generator(p: int) -> int:
        """
        寻找模p的生成元
        对于p = 2q+1，寻找子群的生成元，子群阶为q
        """
        while True:
            h = ChameleonHash.get_random_in_range(p)
            g = pow(h, 2, p)
            if g != 1:
                return pow(g, 2, p)

    @staticmethod
    def crypto_hash(a: int, b: int, bit_length: int) -> int:
        """实现可变长度的密码学哈希函数"""
        # 连接a和b（字节形式）
        x = (a.to_bytes((a.bit_length() + 7) // 8, 'big') +
             b.to_bytes((b.bit_length() + 7) // 8, 'big'))

        # 初始化SHA3
        md = hashlib.sha3_256()
        md.update(x)
        hash_value = int.from_bytes(md.digest(), 'big')

        # 如果哈希长度不足，继续哈希并连接
        while hash_value.bit_length() < bit_length:
            x = (int.from_bytes(x, 'big') + 1).to_bytes((len(x) + 1), 'big')
            md = hashlib.sha3_256()
            md.update(x)
            block = int.from_bytes(md.digest(), 'big')
            hash_value = (hash_value << block.bit_length()) | block

            # 调整为目标长度
        shift_back = hash_value.bit_length() - bit_length
        return hash_value >> shift_back

    @staticmethod
    def key_gen(t: int) -> PrivateKeySet:
        """生成公私钥对"""
        p = ChameleonHash.get_safe_prime(t)
        q = (p - 1) // 2

        g = ChameleonHash.find_generator(p)
        sk = ChameleonHash.get_random_in_range(q)
        pk = pow(g, sk, p)

        return PrivateKeySet(p, q, g, sk, pk)

    @staticmethod
    def hash(data: bytes, rho: int, delta: int, keys: PublicKeySet) -> bytes:
        """计算消息的变色龙哈希值"""
        e = ChameleonHash.crypto_hash(int.from_bytes(data, 'big'), rho, keys.get_p().bit_length())

        t1 = pow(keys.get_public_key(), e, keys.get_p())
        t2 = pow(keys.get_g(), delta, keys.get_p())
        ch = (rho - (t1 * t2) % keys.get_p()) % keys.get_q()

        return ch.to_bytes((ch.bit_length() + 7) // 8, 'big')

    @staticmethod
    def forge(hash_value: bytes, data: bytes, keys: PrivateKeySet) -> PreImage:
        """为新消息伪造原像"""
        c = int.from_bytes(hash_value, 'big')
        m_prime = int.from_bytes(data, 'big')
        k = ChameleonHash.get_random_in_range(keys.get_q())

        rho_prime = (c + pow(keys.get_g(), k, keys.get_p()) % keys.get_q()) % keys.get_q()
        e_prime = ChameleonHash.crypto_hash(m_prime, rho_prime, keys.get_p().bit_length())
        delta_prime = (k - (e_prime * keys.get_secret_key())) % keys.get_q()

        return PreImage(data, rho_prime, delta_prime)

    # ====================== CHT节点定义 ======================


class CHTNode:
    """CHT树节点"""

    def __init__(self):
        self.hash_value = None  # 节点哈希值（字节形式）
        self.rho = None  # 随机数rho
        self.delta = None  # 随机数delta
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.parent = None  # 父节点
        self.is_leaf = False  # 是否为叶节点
        self.data = None  # 叶节点数据引用


# ====================== 模型验证树 ======================

class ModelVerificationTree:
    """支持按模型ID验证的变色龙哈希树"""

    def __init__(self, keys: PrivateKeySet, security_param: int = 512):
        """初始化模型验证树"""
        self.keys = keys
        self.security_param = security_param
        self.public_keys = keys.get_public_key_set()
        self.root = None
        self.model_trees = {}  # 每个模型的子树 {model_id_str: model_root_node}
        self.model_params = {}  # 模型参数映射 {model_id_str: {param_id_str: leaf_node}}
        self.signature = None  # 根节点签名
        self.timestamp = None
        self.version = None

    def build_tree(self, all_models_data: Dict[str, Dict[str, bytes]],
                   signing_key: ecdsa.SigningKey) -> CHTNode:
        """构建模型验证树

        先为每个模型构建子树，再将所有模型子树合并为全局树
        """
        print(f"构建模型验证树，包含 {len(all_models_data)} 个模型")

        # 第一步：为每个模型单独构建子树
        model_roots = []

        # 对模型ID进行排序（按字母顺序），确保构建顺序一致
        for model_id_str in sorted(all_models_data.keys()):
            print(f"  构建模型 {model_id_str} 子树:")
            model_params = all_models_data[model_id_str]

            # 创建该模型的所有叶节点
            leaf_nodes = []
            param_map = {}

            # 对参数ID进行排序
            for param_id_str in sorted(model_params.keys()):
                # 创建叶节点
                node = CHTNode()
                node.is_leaf = True
                node.data = model_params[param_id_str]  # 原始参数数据

                # 编码参数信息用于哈希计算
                encoded_data = self._encode_param(model_id_str, param_id_str, model_params[param_id_str])

                # 生成随机数
                node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                # 计算叶节点哈希
                node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                # 保存叶节点
                leaf_nodes.append(node)
                param_map[param_id_str] = node  # 使用字符串键

                hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])
                print(f"    参数{param_id_str}叶节点：哈希值 = 0x{hash_str}...")

                # 构建该模型的子树
            model_root = self._build_internal_nodes(leaf_nodes)
            model_roots.append(model_root)

            # 保存模型子树和参数映射 - 使用字符串键
            self.model_trees[model_id_str] = model_root
            self.model_params[model_id_str] = param_map

            hash_str = ''.join(f'{b:02x}' for b in model_root.hash_value[:4])
            print(f"  模型 {model_id_str} 子树构建完成，根哈希: 0x{hash_str}...")

            # 第二步：将所有模型的子树合并为全局树
        self.root = self._build_internal_nodes(model_roots)

        # 为根节点添加签名
        self.timestamp = int(time.time())
        self.version = 1
        root_hash_hex = ''.join(f'{b:02x}' for b in self.root.hash_value)
        message = f"{root_hash_hex}|{self.timestamp}|{self.version}".encode()
        self.signature = signing_key.sign(message, hashfunc=hashlib.sha256)

        print(f"全局树构建完成，根哈希: 0x{''.join(f'{b:02x}' for b in self.root.hash_value[:8])}...")

        return self.root

    def _encode_param(self, model_id: str, param_id: str, data: bytes) -> bytes:
        """编码模型参数，用于哈希计算

        现在使用字符串模型ID和参数ID，通过将它们转换为字节并连接来创建唯一编码
        """
        # 将字符串转换为UTF-8字节
        model_bytes = model_id.encode('utf-8')
        param_bytes = param_id.encode('utf-8')

        # 添加长度前缀以确保唯一解码（避免"a1"+"b2"与"a"+"1b2"混淆）
        model_len = len(model_bytes).to_bytes(2, byteorder='big')
        param_len = len(param_bytes).to_bytes(2, byteorder='big')

        return model_len + model_bytes + param_len + param_bytes + data

    def _build_internal_nodes(self, nodes: List[CHTNode]) -> CHTNode:
        """递归构建内部节点"""
        if len(nodes) == 1:
            return nodes[0]

        parent_nodes = []

        # 成对创建父节点
        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # 如果右节点存在
            if i + 1 < len(nodes):
                right_node = nodes[i + 1]

                # 构建父节点
                parent = CHTNode()
                parent.left = left_node
                parent.right = right_node
                left_node.parent = parent
                right_node.parent = parent

                # 连接左右节点哈希值
                combined_data = left_node.hash_value + right_node.hash_value

                parent.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                parent.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                parent.hash_value = ChameleonHash.hash(combined_data, parent.rho, parent.delta, self.public_keys)

                parent_nodes.append(parent)
            else:
                # 当节点数为奇数时，直接提升最后一个节点
                parent_nodes.append(left_node)

                # 递归构建上层节点
        return self._build_internal_nodes(parent_nodes)

    def get_model_proof(self, model_id_str: str) -> Dict[str, Any]:
        """获取整个模型的证明包

        返回该模型的子树根到全局根的证明路径
        """
        if model_id_str not in self.model_trees:
            raise ValueError(f"模型{model_id_str}不存在")

        model_root = self.model_trees[model_id_str]
        model_params = self.model_params[model_id_str]

        # 构建从模型子树根到全局根的证明路径
        proof_path = []
        current = model_root

        while current.parent is not None:
            # 确定当前节点是左节点还是右节点
            is_left = current == current.parent.left

            # 获取兄弟节点
            sibling = current.parent.right if is_left else current.parent.left

            if sibling:
                proof_path.append({
                    'position': 'left' if not is_left else 'right',
                    'hash': sibling.hash_value,
                    'rho': current.parent.rho,
                    'delta': current.parent.delta
                })

            current = current.parent

            # 构建完整的模型验证包
        params_data = {}
        params_proofs = {}

        for param_id_str, leaf_node in sorted(model_params.items()):
            # 获取参数数据
            params_data[param_id_str] = leaf_node.data

            # 获取从叶节点到模型子树根的证明路径
            param_proof = []
            current = leaf_node

            while current.parent is not None and current != model_root:
                # 确定当前节点是左节点还是右节点
                is_left = current == current.parent.left

                # 获取兄弟节点
                sibling = current.parent.right if is_left else current.parent.left

                if sibling:
                    param_proof.append({
                        'position': 'left' if not is_left else 'right',
                        'hash': sibling.hash_value,
                        'rho': current.parent.rho,
                        'delta': current.parent.delta
                    })

                current = current.parent

                # 保存参数的证明路径
            params_proofs[param_id_str] = {
                'rho': leaf_node.rho,
                'delta': leaf_node.delta,
                'proof': param_proof
            }

        return {
            'model_id': model_id_str,  # 直接返回字符串ID
            'params': params_data,
            'params_proofs': params_proofs,
            'model_root_hash': model_root.hash_value,
            'global_proof': proof_path,
            'global_root_hash': self.root.hash_value,
            'timestamp': self.timestamp,
            'version': self.version,
            'signature': self.signature
        }

    def update_param(self, model_id_str: str, param_id_str: str, new_data: bytes) -> bool:
        """更新模型参数并维持CHT完整性"""
        if model_id_str not in self.model_params or param_id_str not in self.model_params[model_id_str]:
            return False

        leaf_node = self.model_params[model_id_str][param_id_str]
        old_data = leaf_node.data
        old_hash = leaf_node.hash_value

        # 创建新的编码数据
        new_encoded_data = self._encode_param(model_id_str, param_id_str, new_data)

        # 寻找新数据的哈希碰撞
        pre_image = ChameleonHash.forge(old_hash, new_encoded_data, self.keys)

        # 更新叶节点
        leaf_node.data = new_data
        leaf_node.rho = pre_image.rho
        leaf_node.delta = pre_image.delta

        # 验证哈希值保持不变
        new_hash = ChameleonHash.hash(new_encoded_data, pre_image.rho, pre_image.delta, self.public_keys)
        if new_hash != old_hash:
            leaf_node.data = old_data  # 恢复原始数据
            return False

        print(f"参数(模型{model_id_str}参数{param_id_str})更新成功，保持哈希值不变")
        return True

    # ====================== 模型验证器 ======================


class ModelVerifier:
    """客户端模型验证器，按模型ID验证整个模型"""

    def __init__(self, ch_public_keys: PublicKeySet, ecdsa_public_key: ecdsa.VerifyingKey):
        """初始化客户端验证器"""
        self.ch_public_keys = ch_public_keys
        self.ecdsa_public_key = ecdsa_public_key
        self.known_model_params = {}  # {model_id_str: {param_id_str: data}}

    def register_model(self, model_id_str: str, params_data: Dict[str, bytes]):
        """注册整个模型的参数，用于后续审计"""
        self.known_model_params[model_id_str] = params_data.copy()

    def _encode_param(self, model_id: str, param_id: str, data: bytes) -> bytes:
        """编码模型参数，与服务器使用相同的方法"""
        # 将字符串转换为UTF-8字节
        model_bytes = model_id.encode('utf-8')
        param_bytes = param_id.encode('utf-8')

        # 添加长度前缀以确保唯一解码
        model_len = len(model_bytes).to_bytes(2, byteorder='big')
        param_len = len(param_bytes).to_bytes(2, byteorder='big')

        return model_len + model_bytes + param_len + param_bytes + data

    def verify_model(self, model_verification_package: Dict) -> Dict[str, Any]:
        """验证整个模型的完整性

        处理流程:
        1. 验证全局根签名
        2. 验证每个参数的局部证明路径
        3. 验证从模型子树到全局根的证明路径
        """
        results = {
            'signature': {'valid': False},
            'params': {},
            'model_path': {'valid': False},
            'overall': {'valid': False}
        }

        pkg = model_verification_package

        # 1. 验证根节点签名
        root_hash_hex = ''.join(f'{b:02x}' for b in pkg['global_root_hash'])
        message = f"{root_hash_hex}|{pkg['timestamp']}|{pkg['version']}".encode()

        try:
            sig_valid = self.ecdsa_public_key.verify(
                pkg['signature'],
                message,
                hashfunc=hashlib.sha256
            )
            results['signature'] = {'valid': sig_valid}

            if not sig_valid:
                results['overall'] = {'valid': False, 'message': "验证失败: 全局签名无效"}
                return results

        except Exception as e:
            results['signature'] = {'valid': False, 'message': f"签名验证异常: {str(e)}"}
            results['overall'] = {'valid': False, 'message': "验证失败: 签名验证异常"}
            return results

            # 2. 验证每个参数的局部证明路径
        params_valid = True
        model_id = pkg['model_id']  # 字符串模型ID

        for param_id_str, param_data in pkg['params'].items():
            # 编码完整参数数据
            encoded_data = self._encode_param(model_id, param_id_str, param_data)

            # 获取参数证明 - 使用字符串键
            param_proof = pkg['params_proofs'][param_id_str]

            # 计算叶节点哈希
            current_hash = ChameleonHash.hash(
                encoded_data,
                param_proof['rho'],
                param_proof['delta'],
                self.ch_public_keys
            )

            # 沿着证明路径计算到模型子树根
            for step in param_proof['proof']:
                sibling_hash = step['hash']

                # 按照位置组合哈希
                if step['position'] == 'left':
                    combined_data = sibling_hash + current_hash
                else:
                    combined_data = current_hash + sibling_hash

                    # 计算父节点哈希
                current_hash = ChameleonHash.hash(
                    combined_data,
                    step['rho'],
                    step['delta'],
                    self.ch_public_keys
                )

                # 检查计算得到的哈希是否与期望的模型根哈希匹配
            param_valid = current_hash == pkg['model_root_hash']
            results['params'][param_id_str] = {'valid': param_valid}

            if not param_valid:
                params_valid = False

        if not params_valid:
            results['overall'] = {'valid': False, 'message': "验证失败: 一个或多个参数验证失败"}
            return results

            # 3. 验证从模型子树到全局根的路径
        current_hash = pkg['model_root_hash']

        for step in pkg['global_proof']:
            sibling_hash = step['hash']

            # 按照位置组合哈希
            if step['position'] == 'left':
                combined_data = sibling_hash + current_hash
            else:
                combined_data = current_hash + sibling_hash

                # 计算父节点哈希
            current_hash = ChameleonHash.hash(
                combined_data,
                step['rho'],
                step['delta'],
                self.ch_public_keys
            )

            # 检查计算得到的根哈希是否与期望的全局根哈希匹配
        model_path_valid = current_hash == pkg['global_root_hash']
        results['model_path'] = {'valid': model_path_valid}

        if not model_path_valid:
            results['overall'] = {'valid': False, 'message': "验证失败: 模型路径验证失败"}
            return results

            # 所有验证都通过
        results['overall'] = {'valid': True, 'message': "验证成功: 所有检查均通过"}
        return results

    def audit_model(self, model_id_str: str, model_params: Dict[str, bytes]) -> List[Dict]:
        """审计模型参数，检测哪些参数被修改"""
        if model_id_str not in self.known_model_params:
            return []

        modified_params = []
        expected_params = self.known_model_params[model_id_str]

        for param_id_str, current_value in model_params.items():
            if param_id_str in expected_params:
                expected_value = expected_params[param_id_str]
                if current_value != expected_value:
                    modified_params.append({
                        'param_id': param_id_str,
                        'original': expected_value,
                        'current': current_value
                    })

        return modified_params

    # ====================== 云服务器实现 ======================


class ModelCloudServer:
    """云服务器实现，支持按model_id获取整个模型"""

    def __init__(self, model_tree: ModelVerificationTree, all_models_data: Dict[str, Dict[str, bytes]]):
        """初始化云服务器"""
        self.model_tree = model_tree
        self.models_data = all_models_data
        self.modified_params = {}  # {model_id_str: [param_id_str, ...]}

    def get_model(self, model_id_str: str, honest: bool = True) -> Dict[str, Any]:
        """获取整个模型及验证所需信息"""
        if model_id_str not in self.models_data:
            raise ValueError(f"模型{model_id_str}不存在")

            # 获取模型验证包
        verification_package = self.model_tree.get_model_proof(model_id_str)

        # 如果不诚实，篡改某个参数但不更新验证信息
        if not honest:
            param_id_str = list(verification_package['params'].keys())[0]  # 选第一个参数篡改
            original_param = verification_package['params'][param_id_str]
            tampered_param = f"{original_param.decode()}_tampered".encode()
            verification_package['params'][param_id_str] = tampered_param
            print(f"云服务器篡改了模型{model_id_str}的参数{param_id_str}:")
            print(f"  原始值: {original_param}")
            print(f"  篡改值: {tampered_param}")

        return verification_package

    def modify_model_param(self, model_id_str: str, param_id_str: str) -> bool:
        """使用变色龙哈希特性修改参数，保持哈希值不变"""
        if model_id_str not in self.models_data or param_id_str not in self.models_data[model_id_str]:
            return False

        original_param = self.models_data[model_id_str][param_id_str]
        modified_param = f"{original_param.decode()}_secretly_modified".encode()

        # 更新模型参数
        success = self.model_tree.update_param(model_id_str, param_id_str, modified_param)

        if success:
            # 更新模型数据
            self.models_data[model_id_str][param_id_str] = modified_param

            # 记录修改
            if model_id_str not in self.modified_params:
                self.modified_params[model_id_str] = []
            self.modified_params[model_id_str].append(param_id_str)

            print(f"云服务器悄悄修改了模型 {model_id_str} 的参数 {param_id_str}:")
            print(f"  原始值: {original_param}")
            print(f"  修改后: {modified_param}")

        return success

    # ====================== 演示程序 ======================


def model_verification_demo():
    """模型验证演示"""
    print("=== 基于模型ID的参数集验证演示 ===\n")

    # 设置随机种子
    random.seed(42)

    # 加载密钥
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    ch_keys = ChameleonHash.key_gen(256)
    print(f"密钥生成完成 (p = {ch_keys.get_p().bit_length()} 位)")

    # 准备模型参数数据 - 使用字符串键，字符串为非数字类型
    model_types = ["cnn1", "resnet18", "transformer"]
    all_models_data = {}

    for i, model_id in enumerate(model_types):
        all_models_data[model_id] = {}
        for param_id in ["weights", "biases", "kernels", "activations"]:
            # 创建唯一参数数据
            all_models_data[model_id][param_id] = f"{model_id}_{param_id}_data".encode()

            # 构建模型验证树
    model_tree = ModelVerificationTree(ch_keys)
    model_tree.build_tree(all_models_data, ecdsa_private_key)

    # 创建云服务器和客户端
    cloud = ModelCloudServer(model_tree, all_models_data)
    client = ModelVerifier(ch_keys.get_public_key_set(), ecdsa_public_key)

    # 客户端注册所有原始模型参数用于审计
    for model_id_str, params in all_models_data.items():
        client.register_model(model_id_str, params)

        # === 演示1: 正常获取模型 ===
    print("\n===== 演示1: 正常获取模型 =====")
    model_id = "cnn1"
    print(f"客户端请求模型{model_id}")

    model_package = cloud.get_model(model_id)

    print(f"收到模型{model_id}，共{len(model_package['params'])}个参数")

    print("\n客户端执行验证:")
    results = client.verify_model(model_package)

    print(f"  签名验证: {'通过' if results['signature']['valid'] else '失败'}")
    print(f"  模型路径验证: {'通过' if results['model_path']['valid'] else '失败'}")
    print(f"  参数验证: {'所有通过' if all(v['valid'] for v in results['params'].values()) else '部分失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    # === 演示2: 检测参数篡改 ===
    print("\n===== 演示2: 检测参数篡改 =====")
    model_id = "resnet18"
    print(f"客户端请求模型{model_id}，但云服务器篡改了一个参数")

    tampered_package = cloud.get_model(model_id, honest=False)

    print("\n客户端执行验证:")
    results = client.verify_model(tampered_package)

    print(f"  签名验证: {'通过' if results['signature']['valid'] else '失败'}")
    print(f"  模型路径验证: {'通过' if results['model_path']['valid'] else '失败'}")

    # 展示每个参数的验证结果
    print("  参数验证结果:")
    for param_id, result in results['params'].items():
        print(f"    参数{param_id}: {'通过' if result['valid'] else '失败'}")

    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    # === 演示3: 合法更新模型参数 ===
    print("\n===== 演示3: 合法更新模型参数 =====")
    model_id, param_id = "transformer", "weights"
    print(f"云服务器合法更新模型{model_id}的参数{param_id}")

    # 云服务器使用变色龙哈希特性更新参数
    cloud.modify_model_param(model_id, param_id)

    # 客户端请求更新后的模型
    print(f"\n客户端请求更新后的模型{model_id}")
    model_package = cloud.get_model(model_id)

    print("\n客户端验证更新后的模型:")
    results = client.verify_model(model_package)

    print(f"  签名验证: {'通过' if results['signature']['valid'] else '失败'}")
    print(f"  模型路径验证: {'通过' if results['model_path']['valid'] else '失败'}")
    print(f"  参数验证: {'所有通过' if all(v['valid'] for v in results['params'].values()) else '部分失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    # === 演示4: 审计模型参数变更 ===
    print("\n===== 演示4: 审计模型参数变更 =====")
    model_id = "transformer"  # 验证刚才更新的模型
    print(f"客户端审计模型{model_id}的参数变更...")

    # 获取当前模型数据
    model_package = cloud.get_model(model_id)

    # 审计变更
    modified_params = client.audit_model(model_id, model_package['params'])

    if modified_params:
        print("\n审计结果 - 发现被修改的参数:")
        for param_info in modified_params:
            param_id = param_info['param_id']
            original = param_info['original']
            current = param_info['current']
            print(f"  参数 {param_id}:")
            print(f"    原始值: {original.hex()[:16]}")
            print(f"    当前值: {current.hex()[:16]}")
    else:
        print("审计未发现任何被修改的参数")

    # === 演示5: 检测非当前请求模型的修改 ===
    print("\n===== 演示5: 检测非当前请求模型的修改 =====")

    # 云服务器秘密修改cnn1模型的kernels参数
    target_model_id, target_param_id = "cnn1", "kernels"
    requested_model_id = "resnet18"  # 客户端将请求不同的模型

    print(f"云服务器利用变色龙哈希特性悄悄修改模型{target_model_id}的参数{target_param_id}")
    cloud.modify_model_param(target_model_id, target_param_id)

    # 客户端请求完全不同的模型，验证应该通过
    print(f"\n客户端请求另一个模型{requested_model_id}")
    response = cloud.get_model(requested_model_id)

    # 客户端验证当前请求的模型
    print("\n客户端执行验证:")
    results = client.verify_model(response)

    print(f"  签名验证: {'通过' if results['signature']['valid'] else '失败'}")
    print(f"  模型路径验证: {'通过' if results['model_path']['valid'] else '失败'}")
    print(f"  参数验证: {'所有通过' if all(v['valid'] for v in results['params'].values()) else '部分失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    print("\n验证通过，但其他模型参数被修改了！现在客户端全面审计所有模型...")

    # 客户端审计所有已知模型
    print("\n客户端执行全局审计，检查所有模型:")

    all_modified = {}
    for audit_model_id in model_types:
        print(f"  审计模型{audit_model_id}...")
        model_package = cloud.get_model(audit_model_id)
        modified = client.audit_model(audit_model_id, model_package['params'])

        if modified:
            all_modified[audit_model_id] = modified

            # 打印审计结果
    if all_modified:
        print("\n全局审计结果:")
        for model_id, params in all_modified.items():
            print(f"  模型 {model_id} 被修改的参数:")
            for param_info in params:
                param_id = param_info['param_id']
                original = param_info['original']
                current = param_info['current']
                print(f"    参数 {param_id}:")
                print(f"      原始值: {original}")
                print(f"      当前值: {current}")
    else:
        print("全局审计未发现任何被修改的参数")

    print("\n=== 模型验证演示完成 ===")


if __name__ == "__main__":
    model_verification_demo()