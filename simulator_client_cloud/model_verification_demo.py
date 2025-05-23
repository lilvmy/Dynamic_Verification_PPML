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
    """Chameleon hash public key set"""

    def __init__(self, p, q, g, pk):
        self.p = p  # Safe prime
        self.q = q  # Prime q, p = 2q + 1
        self.g = g  # Generator
        self.pk = pk  # Public key

    def get_p(self):
        return self.p

    def get_q(self):
        return self.q

    def get_g(self):
        return self.g

    def get_public_key(self):
        return self.pk


class PrivateKeySet(PublicKeySet):
    """Chameleon hash private key set and related parameters"""

    def __init__(self, p, q, g, sk, pk):
        super().__init__(p, q, g, pk)
        self.sk = sk  # Private key

    def get_secret_key(self):
        return self.sk

    def get_public_key_set(self):
        return PublicKeySet(self.p, self.q, self.g, self.pk)

class ModelVerifier:
    """Client model verifier, verifies entire model by model ID"""

    def __init__(self, ch_public_keys: PublicKeySet, ecdsa_public_key: ecdsa.VerifyingKey):
        """Initialize client verifier"""
        self.ch_public_keys = ch_public_keys
        self.ecdsa_public_key = ecdsa_public_key
        self.known_model_params = {}  # {model_id_str: {param_id_str: data}}

    def register_model(self, model_id_str: str, params_data: Dict[str, bytes]):
        """Register entire model parameters for subsequent auditing"""
        self.known_model_params[model_id_str] = params_data.copy()

    def _encode_param(self, model_id: str, param_id: str, data: bytes) -> bytes:
        """Encode model parameters using the same method as server"""
        # Convert strings to UTF-8 bytes
        model_bytes = model_id.encode('utf-8')
        param_bytes = param_id.encode('utf-8')

        # Add length prefix to ensure unique decoding
        model_len = len(model_bytes).to_bytes(2, byteorder='big')
        param_len = len(param_bytes).to_bytes(2, byteorder='big')

        return model_len + model_bytes + param_len + param_bytes + data

    def verify_model(self, model_verification_package: Dict) -> Dict[str, Any]:
        """Verify integrity of entire model

        Processing flow:
        1. Verify global root signature
        2. Verify local proof path for each parameter
        3. Verify proof path from model subtree to global root
        """
        results = {
            'signature': {'valid': False},
            'params': {},
            'model_path': {'valid': False},
            'overall': {'valid': False},
            'timing': {}  # Add timing records
        }

        pkg = model_verification_package

        # 1. Verify root node signature
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
            results['signature'] = {'valid': False, 'message': f"Signature verification exception: {str(e)}"}

        # 2. Verify local proof path for each parameter
        # Use dictionary to record tampered parameters
        tampered_params = {}
        model_id = pkg['model_id']

        # Sort parameters to ensure consistent verification order
        sorted_param_ids = sorted(pkg['params'].keys())

        for param_id_str in sorted_param_ids:
            param_data = pkg['params'][param_id_str]
            # Encode complete parameter data
            encoded_data = self._encode_param(model_id, param_id_str, param_data)

            # Get parameter proof
            param_proof = pkg['params_proofs'][param_id_str]

            # Calculate leaf node hash
            current_hash = ChameleonHash.hash(
                encoded_data,
                param_proof['rho'],
                param_proof['delta'],
                self.ch_public_keys
            )

            # Save original leaf hash for determining if parameter was tampered
            original_leaf_hash = current_hash

            # Calculate along proof path to model subtree root
            for step in param_proof['proof']:
                sibling_hash = step['hash']

                # Combine hash according to position
                if step['position'] == 'left':
                    combined_data = sibling_hash + current_hash
                else:
                    combined_data = current_hash + sibling_hash

                # Calculate parent node hash
                current_hash = ChameleonHash.hash(
                    combined_data,
                    step['rho'],
                    step['delta'],
                    self.ch_public_keys
                )

            # Check if calculated hash matches expected model root hash
            param_valid = current_hash == pkg['model_root_hash']
            results['params'][param_id_str] = {'valid': param_valid}

            # If parameter verification fails, record to tampered list
            if not param_valid:
                tampered_params[param_id_str] = original_leaf_hash

        # Calculate number of successfully verified and failed parameters
        valid_params_count = sum(1 for result in results['params'].values() if result['valid'])
        invalid_params_count = len(results['params']) - valid_params_count

        # 3. Verify path from model subtree to global root
        model_path_valid = len(tampered_params) == 0

        # For completeness, we still execute verification steps
        current_hash = pkg['model_root_hash']

        for step in pkg['global_proof']:
            sibling_hash = step['hash']

            # Combine hash according to position
            if step['position'] == 'left':
                combined_data = sibling_hash + current_hash
            else:
                combined_data = current_hash + sibling_hash

            # Calculate parent node hash
            current_hash = ChameleonHash.hash(
                combined_data,
                step['rho'],
                step['delta'],
                self.ch_public_keys
            )

        # Check if calculated root hash matches expected global root hash
        computed_path_valid = current_hash == pkg['global_root_hash']

        # Model path verification result should reflect impact of parameter tampering
        # If parameters were tampered, even if path verification calculation succeeds, we should mark it as failed
        results['model_path'] = {
            'valid': model_path_valid and computed_path_valid,
            'computed_valid': computed_path_valid,
            'tampered_params_detected': len(tampered_params) > 0
        }

        # Overall verification result
        signature_valid = results['signature']['valid']
        params_all_valid = valid_params_count == len(results['params'])

        # Determine overall verification result
        overall_valid = signature_valid and params_all_valid and results['model_path']['valid']

        if not signature_valid:
            results['overall'] = {'valid': False, 'message': "Verification failed: Global signature invalid"}
        elif not params_all_valid:
            results['overall'] = {'valid': False, 'message': f"Verification failed: {invalid_params_count} parameter verifications failed"}
        elif not results['model_path']['valid']:
            results['overall'] = {'valid': False, 'message': "Verification failed: Model path verification failed"}
        else:
            results['overall'] = {'valid': True, 'message': "Verification successful: All checks passed"}

        # Add detailed information of verification results
        results['summary'] = {
            'Signature verification': 'Success' if signature_valid else 'Failed',
            'Model path verification': 'Success' if results['model_path']['valid'] else 'Failed',
            'Parameter verification': f"{valid_params_count} successful, {invalid_params_count} failed",
            'Overall result': 'Verification successful' if overall_valid else 'Verification failed',
        }

        return results

    def audit_model(self, model_id_str: str, model_params: Dict[str, bytes]) -> List[Dict]:
        """Audit model parameters, detect which parameters were modified"""
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

# ====================== Cloud Server Implementation ======================
class ModelCloudServer:
    """Cloud server implementation, supports getting entire model by model_id"""

    def __init__(self, HE, model_tree: ChameleonHashTree, all_models_data: Dict[str, Dict[str, bytes]]):
        """Initialize cloud server"""
        self.model_tree = model_tree
        self.HE = HE
        self.models_data = all_models_data
        self.modified_params = {}  # {model_id_str: [param_id_str, ...]}

    def get_model(self, model_id_str: str, tamper_param_size = None, honest: bool = True) -> Dict[str, Any]:
        """Get entire model and information required for verification"""
        if model_id_str not in self.models_data:
            raise ValueError(f"Model {model_id_str} does not exist")

        # Get model verification package
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
        """Use chameleon hash property to modify parameters while keeping hash value unchanged

        Supports following calling methods:
        1. modify_model_param("model1", "param1") - Modify single parameter of single model
        2. modify_model_param("model1", ["param1", "param2"]) - Modify multiple parameters of single model
        3. modify_model_param(["model1", "model2"], "param1") - Modify same parameter of multiple models
        4. modify_model_param(["model1", "model2"], {"model1": ["param1", "param2"], "model2": ["param1"]})
           - Modify specific parameters of multiple models
        5. modify_model_param("model1", {"param1": new_value1, "param2": new_value2})
           - Modify specific parameters of single model and provide new parameter values
        """
        # Construct parameter modification mapping
        param_modifications = {}

        # 1. Single model, single parameter
        if isinstance(model_id_str, str) and isinstance(param_id_str, str):
            if model_id_str not in self.models_data or param_id_str not in self.models_data[model_id_str]:
                print(f"Parameter modification failed: Parameter {param_id_str} of model {model_id_str} does not exist")
                return None

            original_param = self.models_data[model_id_str][param_id_str]
            modified_param = hashlib.sha256(original_param + b"_modified").digest()

            param_modifications = {
                model_id_str: {
                    param_id_str: modified_param
                }
            }

        # 2. Single model, multiple parameters (list form)
        elif isinstance(model_id_str, str) and isinstance(param_id_str, list):
            if model_id_str not in self.models_data:
                print(f"Parameter modification failed: Model {model_id_str} does not exist")
                return None

            param_dict = {}
            for param_id in param_id_str:
                if param_id not in self.models_data[model_id_str]:
                    print(f"Warning: Parameter {param_id} of model {model_id_str} does not exist, skipped")
                    continue

                original_param = self.models_data[model_id_str][param_id]
                modified_param = hashlib.sha256(original_param + b"_modified").digest()
                param_dict[param_id] = modified_param

            if param_dict:
                param_modifications = {model_id_str: param_dict}
            else:
                print(f"Parameter modification failed: No valid parameters to modify")
                return None

        # 5. Single model, multiple parameters (dictionary form, supports custom values)
        elif isinstance(model_id_str, str) and isinstance(param_id_str, dict):
            if model_id_str not in self.models_data:
                print(f"Parameter modification failed: Model {model_id_str} does not exist")
                return None

            param_dict = {}
            for param_id, new_value in param_id_str.items():
                if param_id not in self.models_data[model_id_str]:
                    print(f"Warning: Parameter {param_id} of model {model_id_str} does not exist, skipped")
                    continue

                # If new parameter value is provided, use it directly; otherwise use default modification method
                if new_value is not None:
                    if not isinstance(new_value, bytes):
                        print(f"Warning: New value for parameter {param_id} must be bytes type, skipped")
                        continue
                    modified_param = new_value
                else:
                    original_param = self.models_data[model_id_str][param_id]
                    modified_param = hashlib.sha256(original_param + b"_modified").digest()

                param_dict[param_id] = modified_param

            if param_dict:
                param_modifications = {model_id_str: param_dict}
            else:
                print(f"Parameter modification failed: No valid parameters to modify")
                return None

        # 3. Multiple models, single same parameter
        elif isinstance(model_id_str, list) and isinstance(param_id_str, str):
            for model_id in model_id_str:
                if model_id not in self.models_data:
                    print(f"Warning: Model {model_id} does not exist, skipped")
                    continue

                if param_id_str not in self.models_data[model_id]:
                    print(f"Warning: Parameter {param_id_str} of model {model_id} does not exist, skipped")
                    continue

                original_param = self.models_data[model_id][param_id_str]
                modified_param = hashlib.sha256(original_param + b"_modified").digest()

                if model_id not in param_modifications:
                    param_modifications[model_id] = {}

                param_modifications[model_id][param_id_str] = modified_param

            if not param_modifications:
                print(f"Parameter modification failed: No valid models and parameters to modify")
                return None

        # 4. Multiple models, each model specifies parameters
        elif isinstance(model_id_str, list) and isinstance(param_id_str, dict):
            for model_id in model_id_str:
                if model_id not in self.models_data:
                    print(f"Warning: Model {model_id} does not exist, skipped")
                    continue

                if model_id not in param_id_str:
                    print(f"Warning: No parameters specified for model {model_id}, skipped")
                    continue

                model_params = param_id_str[model_id]
                if not isinstance(model_params, list):
                    print(f"Warning: Parameters for model {model_id} must be a list, skipped")
                    continue

                param_dict = {}
                for param_id in model_params:
                    if param_id not in self.models_data[model_id]:
                        print(f"Warning: Parameter {param_id} of model {model_id} does not exist, skipped")
                        continue

                    original_param = self.models_data[model_id][param_id]
                    modified_param = hashlib.sha256(original_param + b"_modified").digest()
                    param_dict[param_id] = modified_param

                if param_dict:
                    param_modifications[model_id] = param_dict

            if not param_modifications:
                print(f"Parameter modification failed: No valid models and parameters to modify")
                return None
        else:
            print(f"Parameter error: Unsupported parameter combination")
            return None

        # Call update_model_or_params method
        result_tree = self.model_tree.update_model_or_params(
            param_modifications=param_modifications
        )

        # Update local model data
        for model_id, params in param_modifications.items():
            for param_id, new_value in params.items():
                # Only update parameters that exist in model data
                if model_id in self.models_data and param_id in self.models_data[model_id]:
                    self.models_data[model_id][param_id] = new_value

                    # Record modification
                    if model_id not in self.modified_params:
                        self.modified_params[model_id] = []
                    if param_id not in self.modified_params[model_id]:
                        self.modified_params[model_id].append(param_id)

        print(f"Parameter modification successful: {sum(len(params) for params in param_modifications.values())} parameters modified")
        return result_tree

    def add_new_model(self, model_id_str: Union[str, List[str]],model_params: Union[Dict[str, bytes], Dict[str, Dict[str, bytes]]]) -> ChameleonHashTree:
        """Add new model to system, supports single model or batch adding multiple models"""
        # Handle cases of single model and batch models
        if isinstance(model_id_str, str):
            # Single model case
            if model_id_str in self.models_data:
                print(f"Add model failed: Model {model_id_str} already exists")
                return None

            # Construct add model request
            model_to_add = {
                model_id_str: model_params
            }
        else:
            # Model ID list case
            # Verify if all model IDs already exist
            existing_models = []
            for model_id in model_id_str:
                if model_id in self.models_data:
                    existing_models.append(model_id)

            if existing_models:
                print(f"Add model failed: Following models already exist: {', '.join(existing_models)}")
                return None

            # If model_params is single-layer dictionary while model_id_str is list, this is mismatched
            if not isinstance(next(iter(model_params.values()), {}), dict):
                print(f"Parameter error: When providing multiple model IDs, model_params must be nested dictionary {model_id: {model_params.keys(): data, ...}, ...}")
                return None

            # Check if model_params contains all models to be added
            missing_models = set(model_id_str) - set(model_params.keys())
            if missing_models:
                print(f"Parameter error: Parameters not provided for following model IDs: {', '.join(missing_models)}")
                return None

            # Extract model parameters to be added
            model_to_add = {model_id: model_params[model_id] for model_id in model_id_str}

        # Call update_model_or_params method and modify return value handling
        tree = self.model_tree.update_model_or_params(
            model_to_add=model_to_add
        )

        return tree

    def delete_model(self, model_id_str: Union[str, List[str]]) -> ChameleonHashTree:
        """Delete model from system, supports single model or batch deleting multiple models"""
        # Handle cases of single model and batch models
        if isinstance(model_id_str, str):
            # Single model case
            if model_id_str not in self.models_data:
                print(f"Delete model failed: Model {model_id_str} does not exist")
                return False

            models_to_delete = [model_id_str]
        else:
            # Model ID list case
            # Verify if all model IDs exist
            non_existing_models = []
            for model_id in model_id_str:
                if model_id not in self.models_data:
                    non_existing_models.append(model_id)

            if non_existing_models:
                print(f"Delete model failed: Following models do not exist: {', '.join(non_existing_models)}")
                return False

            models_to_delete = model_id_str

        # Delete models one by one
        all_success = True
        for model_id in models_to_delete:
            # Call update_model_or_params method
            root_node = self.model_tree.update_model_or_params(
                model_id_to_delete=model_id
            )

        return root_node

# ====================== Demo Program ======================

def main():
    """
    Model verification display
    """
    print("=====model verification based on model id======\n")
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

    # get encrypted model params
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

    # === Demo 5: Detect modification of non-current requested model ===
    print("\n===== Demo 5: Detect modification of non-current requested model =====")

    # Cloud server secretly modifies bias parameter of cnn1 model
    target_model_id, target_param_id = "cnn1", "cnn1.0.bias"
    requested_model_id = "cnn2"  # Client will request different model

    print(f"Cloud server uses chameleon hash property to secretly modify parameter {target_param_id} of model {target_model_id}")
    cloud.modify_model_param(target_model_id, target_param_id)

    # Client requests completely different model, verification should pass
    print(f"\nClient requests another model {requested_model_id}")
    response = cloud.get_model(requested_model_id)

    # Client verifies currently requested model
    print("\nClient performs verification:")
    results = client.verify_model(response)

    print(f"  Signature verification: {'Pass' if results['signature']['valid'] else 'Failed'}")
    print(f"  Model path verification: {'Pass' if results['model_path']['valid'] else 'Failed'}")
    print(f"  Parameter verification: {'All pass' if all(v['valid'] for v in results['params'].values()) else 'Partial failure'}")
    print(f"  Overall result: {'Verification passed' if results['overall']['valid'] else 'Verification failed'}")

    print("\nVerification passed, but other model parameters were modified! Now client performs comprehensive audit of all models...")

    # Client audits all known models
    print("\nClient performs global audit, checking all models:")

    all_modified = {}
    for audit_model_id in model_id_mapping.keys():
        print(f"  Auditing model {audit_model_id}...")
        model_package = cloud.get_model(audit_model_id)
        modified = client.audit_model(audit_model_id, model_package['params'])

        if modified:
            all_modified[audit_model_id] = modified

    # Print audit results
    if all_modified:
        print("\nGlobal audit results:")
        for model_id, params in all_modified.items():
            print(f"  Model {model_id} modified parameters:")
            for param_info in params:
                param_id = param_info['param_id']
                original = param_info['original']
                current = param_info['current']
                print(f"    Parameter {param_id}:")
                print(f"      Original value: {original}")
                print(f"      Current value: {current}")
    else:
        print("Global audit found no modified parameters")

    print("\n=== Model verification demo completed ===")


if __name__ == "__main__":
    main()
