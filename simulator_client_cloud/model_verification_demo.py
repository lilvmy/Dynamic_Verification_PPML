import random
from initialization.setup import load_ecdsa_keys
from dual_verification_tree.CHT_utils import load_cht_keys, ChameleonHash
import ecdsa
from typing import List, Dict, Any, Union
import hashlib
from dual_verification_tree.build_CHT import CHTNode, ChameleonHashTree, load_chameleon_hash_tree
from level_homomorphic_encryption.encrypted_process_model import binary_mean_representation, extract_param_features, extract_data_from_hash_node
from initialization.setup import load_HE_keys
import time
import numpy as np
from utils.util import generate_random_matrices
import pickle


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
            'overall': {'valid': False},
            'timing': {}  # 添加时间记录
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
        except Exception as e:
            results['signature'] = {'valid': False, 'message': f"签名验证异常: {str(e)}"}

        # 2. 验证每个参数的局部证明路径
        # 使用字典记录篡改的参数
        tampered_params = {}
        model_id = pkg['model_id']

        # 将参数排序以确保每次验证顺序一致
        sorted_param_ids = sorted(pkg['params'].keys())

        for param_id_str in sorted_param_ids:
            param_data = pkg['params'][param_id_str]
            # 编码完整参数数据
            encoded_data = self._encode_param(model_id, param_id_str, param_data)

            # 获取参数证明
            param_proof = pkg['params_proofs'][param_id_str]

            # 计算叶节点哈希
            current_hash = ChameleonHash.hash(
                encoded_data,
                param_proof['rho'],
                param_proof['delta'],
                self.ch_public_keys
            )

            # 保存原始叶子哈希，用于判断参数是否被篡改
            original_leaf_hash = current_hash

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

            # 如果参数验证失败，记录到篡改列表
            if not param_valid:
                tampered_params[param_id_str] = original_leaf_hash

        # 计算成功验证的参数数量和失败的参数数量
        valid_params_count = sum(1 for result in results['params'].values() if result['valid'])
        invalid_params_count = len(results['params']) - valid_params_count

        # 3. 验证从模型子树到全局根的路径
        model_path_valid = len(tampered_params) == 0

        # 为了完整性，我们仍然执行验证步骤
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
        computed_path_valid = current_hash == pkg['global_root_hash']

        # 模型路径验证结果应该反映参数篡改的影响
        # 如果有参数被篡改，即使路径验证计算成功，我们也应将其标记为失败
        results['model_path'] = {
            'valid': model_path_valid and computed_path_valid,
            'computed_valid': computed_path_valid,
            'tampered_params_detected': len(tampered_params) > 0
        }

        # 整体验证结果
        signature_valid = results['signature']['valid']
        params_all_valid = valid_params_count == len(results['params'])

        # 决定整体验证结果
        overall_valid = signature_valid and params_all_valid and results['model_path']['valid']

        if not signature_valid:
            results['overall'] = {'valid': False, 'message': "验证失败: 全局签名无效"}
        elif not params_all_valid:
            results['overall'] = {'valid': False, 'message': f"验证失败: {invalid_params_count}个参数验证失败"}
        elif not results['model_path']['valid']:
            results['overall'] = {'valid': False, 'message': "验证失败: 模型路径验证失败"}
        else:
            results['overall'] = {'valid': True, 'message': "验证成功: 所有检查均通过"}


        # 添加验证结果的详细信息
        results['summary'] = {
            '验证签名': '成功' if signature_valid else '失败',
            '验证模型路径': '成功' if results['model_path']['valid'] else '失败',
            '参数验证': f"{valid_params_count} 个成功, {invalid_params_count} 个失败",
            '整体结果': '验证成功' if overall_valid else '验证失败',
        }

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

    def __init__(self, HE, model_tree: ChameleonHashTree, all_models_data: Dict[str, Dict[str, bytes]]):
        """初始化云服务器"""
        self.model_tree = model_tree
        self.HE = HE
        self.models_data = all_models_data
        self.modified_params = {}  # {model_id_str: [param_id_str, ...]}

    def get_model(self, model_id_str: str, tamper_param_size = None, honest: bool = True) -> Dict[str, Any]:
        """获取整个模型及验证所需信息"""
        if model_id_str not in self.models_data:
            raise ValueError(f"模型{model_id_str}不存在")

        # 获取模型验证包
        verification_package = self.model_tree.get_model_proof(model_id_str)

        # if malicious, tamper param
        if not honest:
            all_params_keys = list(verification_package['params'].keys())

            param_id_str = all_params_keys[0:tamper_param_size]

            for i in range(len(param_id_str)):
                # save original hash of param_id
                original_hash = verification_package['params'][param_id_str[i]]
                # build a different hash to simulate tamper
                if param_id_str[i].endswith('weight'):
                    tampered_param = generate_random_matrices(count=64, fixed_shape=(3, 3), min_val=-1, max_val=1)
                else:
                    tampered_param = generate_random_matrices(count=1, fixed_shape=(1, 64), min_val=-1, max_val=1)
                tampered_binary_info = binary_mean_representation(tampered_param)
                tampered_binary_mean = tampered_binary_info['binary_mean']
                # print(f"  Shape: {tampered_binary_info['shape']}, Elements: {tampered_param.size}")
                # print(f"  Binary mean: {tampered_binary_mean:.4f}")
                # print(f"  Positive elements mean: {tampered_binary_info['positive_mean']:.4f}")
                # print(f"  Negative elements mean: {tampered_binary_info['negative_mean']:.4f}")
                tampered_param_features = extract_param_features(tampered_param, param_id_str[i])
                tampered_param_features_bytes = pickle.dumps(tampered_param_features)
                encrypted_tampered_param_mean = self.HE.encode_number(tampered_binary_mean)
                encrypted_tampered_param_mean_bytes = encrypted_tampered_param_mean.to_bytes()
                tampered_param_feature_length = len(tampered_param_features_bytes)

                final_tampered_param = tampered_param_feature_length.to_bytes(4, byteorder='big') + tampered_param_features_bytes + encrypted_tampered_param_mean_bytes

                # tampered_hash = hashlib.sha256(
                #     original_hash + final_tampered_param
                # ).digest()
                verification_package['params'][param_id_str[i]] = final_tampered_param
                # print(f"the cloud server tamper model {model_id_str}'s parameter {param_id_str[i]}:")
                # print(f"  original hash: {original_hash.hex()[:16]}...")
                # print(f"  tamper hash: {final_tampered_param.hex()[:16]}...")

        return verification_package

    def modify_model_param(self, model_id_str: Union[str, List[str]],param_id_str: Union[str, List[str], Dict[str, List[str]]]) -> ChameleonHashTree:
        """使用变色龙哈希特性修改参数，保持哈希值不变

        支持以下调用方式:
        1. modify_model_param("model1", "param1") - 修改单个模型的单个参数
        2. modify_model_param("model1", ["param1", "param2"]) - 修改单个模型的多个参数
        3. modify_model_param(["model1", "model2"], "param1") - 修改多个模型的相同参数
        4. modify_model_param(["model1", "model2"], {"model1": ["param1", "param2"], "model2": ["param1"]})
           - 修改多个模型的特定参数
        5. modify_model_param("model1", {"param1": new_value1, "param2": new_value2})
           - 修改单个模型的特定参数，并提供新的参数值
        """
        # 构造参数修改映射
        param_modifications = {}

        # 1. 单个模型，单个参数
        if isinstance(model_id_str, str) and isinstance(param_id_str, str):
            if model_id_str not in self.models_data or param_id_str not in self.models_data[model_id_str]:
                print(f"修改参数失败: 模型 {model_id_str} 的参数 {param_id_str} 不存在")
                return None

            original_param = self.models_data[model_id_str][param_id_str]
            modified_param = hashlib.sha256(original_param + b"_modified").digest()

            param_modifications = {
                model_id_str: {
                    param_id_str: modified_param
                }
            }

            # 2. 单个模型，多个参数（列表形式）
        elif isinstance(model_id_str, str) and isinstance(param_id_str, list):
            if model_id_str not in self.models_data:
                print(f"修改参数失败: 模型 {model_id_str} 不存在")
                return None

            param_dict = {}
            for param_id in param_id_str:
                if param_id not in self.models_data[model_id_str]:
                    print(f"警告: 模型 {model_id_str} 的参数 {param_id} 不存在，已跳过")
                    continue

                original_param = self.models_data[model_id_str][param_id]
                modified_param = hashlib.sha256(original_param + b"_modified").digest()
                param_dict[param_id] = modified_param

            if param_dict:
                param_modifications = {model_id_str: param_dict}
            else:
                print(f"修改参数失败: 没有有效的参数可以修改")
                return None

                # 5. 单个模型，多个参数（字典形式，支持自定义值）
        elif isinstance(model_id_str, str) and isinstance(param_id_str, dict):
            if model_id_str not in self.models_data:
                print(f"修改参数失败: 模型 {model_id_str} 不存在")
                return None

            param_dict = {}
            for param_id, new_value in param_id_str.items():
                if param_id not in self.models_data[model_id_str]:
                    print(f"警告: 模型 {model_id_str} 的参数 {param_id} 不存在，已跳过")
                    continue

                    # 如果提供了新的参数值，直接使用它；否则使用默认的修改方式
                if new_value is not None:
                    if not isinstance(new_value, bytes):
                        print(f"警告: 参数 {param_id} 的新值必须是bytes类型，已跳过")
                        continue
                    modified_param = new_value
                else:
                    original_param = self.models_data[model_id_str][param_id]
                    modified_param = hashlib.sha256(original_param + b"_modified").digest()

                param_dict[param_id] = modified_param

            if param_dict:
                param_modifications = {model_id_str: param_dict}
            else:
                print(f"修改参数失败: 没有有效的参数可以修改")
                return None

                # 3. 多个模型，单个相同参数
        elif isinstance(model_id_str, list) and isinstance(param_id_str, str):
            for model_id in model_id_str:
                if model_id not in self.models_data:
                    print(f"警告: 模型 {model_id} 不存在，已跳过")
                    continue

                if param_id_str not in self.models_data[model_id]:
                    print(f"警告: 模型 {model_id} 的参数 {param_id_str} 不存在，已跳过")
                    continue

                original_param = self.models_data[model_id][param_id_str]
                modified_param = hashlib.sha256(original_param + b"_modified").digest()

                if model_id not in param_modifications:
                    param_modifications[model_id] = {}

                param_modifications[model_id][param_id_str] = modified_param

            if not param_modifications:
                print(f"修改参数失败: 没有有效的模型和参数可以修改")
                return None

                # 4. 多个模型，每个模型指定参数
        elif isinstance(model_id_str, list) and isinstance(param_id_str, dict):
            for model_id in model_id_str:
                if model_id not in self.models_data:
                    print(f"警告: 模型 {model_id} 不存在，已跳过")
                    continue

                if model_id not in param_id_str:
                    print(f"警告: 未为模型 {model_id} 指定参数，已跳过")
                    continue

                model_params = param_id_str[model_id]
                if not isinstance(model_params, list):
                    print(f"警告: 模型 {model_id} 的参数必须是列表，已跳过")
                    continue

                param_dict = {}
                for param_id in model_params:
                    if param_id not in self.models_data[model_id]:
                        print(f"警告: 模型 {model_id} 的参数 {param_id} 不存在，已跳过")
                        continue

                    original_param = self.models_data[model_id][param_id]
                    modified_param = hashlib.sha256(original_param + b"_modified").digest()
                    param_dict[param_id] = modified_param

                if param_dict:
                    param_modifications[model_id] = param_dict

            if not param_modifications:
                print(f"修改参数失败: 没有有效的模型和参数可以修改")
                return None
        else:
            print(f"参数错误: 不支持的参数组合")
            return None

            # 调用update_model_or_params方法
        result_tree = self.model_tree.update_model_or_params(
            param_modifications=param_modifications
        )

        # 更新本地模型数据
        for model_id, params in param_modifications.items():
            for param_id, new_value in params.items():
                # 只更新模型数据中存在的参数
                if model_id in self.models_data and param_id in self.models_data[model_id]:
                    self.models_data[model_id][param_id] = new_value

                    # 记录修改
                    if model_id not in self.modified_params:
                        self.modified_params[model_id] = []
                    if param_id not in self.modified_params[model_id]:
                        self.modified_params[model_id].append(param_id)

        print(f"参数修改成功: {sum(len(params) for params in param_modifications.values())} 个参数已修改")
        return result_tree

    def add_new_model(self, model_id_str: Union[str, List[str]],model_params: Union[Dict[str, bytes], Dict[str, Dict[str, bytes]]]) -> ChameleonHashTree:
        """添加新模型到系统中，支持单个模型或批量添加多个模型"""
        # 处理单个模型和批量模型的情况
        if isinstance(model_id_str, str):
            # 单个模型的情况
            if model_id_str in self.models_data:
                print(f"添加模型失败: 模型 {model_id_str} 已存在")
                return None

                # 构造添加模型请求
            model_to_add = {
                model_id_str: model_params
            }
        else:
            # 模型ID列表的情况
            # 验证所有模型ID是否已存在
            existing_models = []
            for model_id in model_id_str:
                if model_id in self.models_data:
                    existing_models.append(model_id)

            if existing_models:
                print(f"添加模型失败: 以下模型已存在: {', '.join(existing_models)}")
                return None

                # 如果model_params是单层字典，而model_id_str是列表，这是不匹配的
            if not isinstance(next(iter(model_params.values()), {}), dict):
                print(f"参数错误: 当提供多个模型ID时，model_params必须是嵌套字典 {model_id: {model_params.keys(): data, ...}, ...}")
                return None

                # 检查提供的model_params中是否包含了所有需要添加的模型
            missing_models = set(model_id_str) - set(model_params.keys())
            if missing_models:
                print(f"参数错误: 以下模型ID的参数未提供: {', '.join(missing_models)}")
                return None

                # 提取需要添加的模型参数
            model_to_add = {model_id: model_params[model_id] for model_id in model_id_str}

        # 调用update_model_or_params方法，并修改返回值处理
        tree = self.model_tree.update_model_or_params(
            model_to_add=model_to_add
        )

        return tree


    def delete_model(self, model_id_str: Union[str, List[str]]) -> ChameleonHashTree:
        """从系统中删除模型，支持单个模型或批量删除多个模型"""
        # 处理单个模型和批量模型的情况
        if isinstance(model_id_str, str):
            # 单个模型的情况
            if model_id_str not in self.models_data:
                print(f"删除模型失败: 模型 {model_id_str} 不存在")
                return False

            models_to_delete = [model_id_str]
        else:
            # 模型ID列表的情况
            # 验证所有模型ID是否存在
            non_existing_models = []
            for model_id in model_id_str:
                if model_id not in self.models_data:
                    non_existing_models.append(model_id)

            if non_existing_models:
                print(f"删除模型失败: 以下模型不存在: {', '.join(non_existing_models)}")
                return False

            models_to_delete = model_id_str

        # 逐个删除模型
        all_success = True
        for model_id in models_to_delete:
            # 调用update_model_or_params方法
            root_node = self.model_tree.update_model_or_params(
                model_id_to_delete=model_id
            )

        return root_node

    # ====================== 演示程序 ======================

def main():
    """
    model verification display
    """
    print("=====model veritication based on model id======\n")
    # set random seeds
    random.seed(42)

    # load signature keys, cht_keys_params, HE keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)
    HE = load_HE_keys()


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
    CHT = load_chameleon_hash_tree("../dual_verification_tree/tree/CHT_10.tree")
    print(f"CHT load successfully {CHT}")

    # create cloud server and client
    cloud = ModelCloudServer(HE, CHT, all_models_data)
    client = ModelVerifier(cht_keys.get_public_key_set(), ecdsa_public_key)

    # client register all of param of the model to audit
    for model_id_str, params in all_models_data.items():
        client.register_model(model_id_str, params)


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