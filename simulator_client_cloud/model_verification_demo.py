import random
from initialization.setup import load_ecdsa_keys
from dual_verification_tree.CHT_utils import load_cht_keys, ChameleonHash
import ecdsa
from typing import List, Dict, Any
import hashlib
from dual_verification_tree.build_CHT import CHTNode, ChameleonHashTree, load_chameleon_hash_tree
from level_homomorphic_encryption.encrypted_process_model import extract_data_from_hash_node
import time

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

    def __init__(self, model_tree: ChameleonHashTree, all_models_data: Dict[str, Dict[str, bytes]]):
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
            # 创建一个不同的哈希值来模拟篡改
            tampered_hash = hashlib.sha256(
                verification_package['params'][param_id_str] + b"tampered"
            ).digest()
            verification_package['params'][param_id_str] = tampered_hash
            print(f"云服务器篡改了模型{model_id_str}的参数{param_id_str}:")
            print(f"  原始哈希: {verification_package['params'][param_id_str].hex()[:16]}...")
            print(f"  篡改哈希: {tampered_hash.hex()[:16]}...")

        return verification_package

    def modify_model_param(self, model_id_str: str, param_id_str: str) -> bool:
        """使用变色龙哈希特性修改参数，保持哈希值不变"""
        if model_id_str not in self.models_data or param_id_str not in self.models_data[model_id_str]:
            return False

        original_param = self.models_data[model_id_str][param_id_str]
        # 创建一个不同的哈希值，但保持固定长度
        modified_param = hashlib.sha256(original_param + b"_modified").digest()

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

def main():
    """
    model verification display
    """
    print("=====model veritication based on model id======\n")
    # set random seeds
    random.seed(42)

    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    # load cht_keys_params
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)

    all_models_data = {}
    model_id_mapping = {}
    # get model id
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/model_id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    # # get encrypted model params
    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        encrypted_model_param = extract_data_from_hash_node(encrypted_path)
        for name, param in encrypted_model_param.items():
            all_models_data[model_id][name] = param

    # load CHT
    CHT = load_chameleon_hash_tree("../dual_verification_tree/CHT.tree")
    print(f"CHT load successfully {CHT}")

    # create cloud server and client
    cloud = ModelCloudServer(CHT, all_models_data)
    client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)

    # client register all of param of the model to audit
    for model_id_str, params in all_models_data.items():
        client.register_model(model_id_str, params)

    # === 演示1: 正常获取模型 ===
    print("\n===== 演示1: 正常获取模型 =====")
    start_time = time.time()
    model_id = "cnn3"
    print(f"客户端请求模型{model_id}")

    model_package = cloud.get_model(model_id)

    print(f"收到模型{model_id}, 共{len(model_package['params'])}个参数")

    print("\n客户端执行验证:")
    results = client.verify_model(model_package)
    end_time = time.time()

    print(f"   验证时间为： {end_time - start_time}")
    print(f"  签名验证: {'通过' if results['signature']['valid'] else '失败'}")
    print(f"  模型路径验证: {'通过' if results['model_path']['valid'] else '失败'}")
    print(f"  参数验证: {'所有通过' if all(v['valid'] for v in results['params'].values()) else '部分失败'}")
    print(f"  总体结果: {'验证通过' if results['overall']['valid'] else '验证失败'}")

    # === 演示2: 检测参数篡改 ===
    print("\n===== 演示2: 检测参数篡改 =====")
    model_id = "cnn1"
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
    model_id, param_id = "cnn2", "cnn2.0.weights"
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
    model_id = "cnn2"  # 验证刚才更新的模型
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

        # 云服务器秘密修改cnn1模型的bias参数
        target_model_id, target_param_id = "cnn1", "cnn1.0.bias"
        requested_model_id = "cnn2"  # 客户端将请求不同的模型

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
        for audit_model_id in model_id_mapping.keys():
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
    main()