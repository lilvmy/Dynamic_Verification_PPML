
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
    """Merkle Hash Tree Node"""

    def __init__(self):
        """Initialize MHT Node"""
        self.hash_value = None  # Node hash value
        self.is_leaf = False  # Whether it's a leaf node
        self.data = None  # Leaf node data
        self.left = None  # Left child node
        self.right = None  # Right child node
        self.parent = None  # Parent node
        self.secure_code = None
        # For visualization and debugging
        self.param_id = None  # Parameter ID (for leaf nodes)
        self.model_id = None  # Model ID (for model subtree root nodes)
        self.block_idx = None  # Index of parameter block


class MerkleHashTree:
    """SHA-256 based Merkle Hash Tree implementation"""

    def __init__(self):
        """Initialize Merkle Hash Tree"""
        self.root = None
        self.model_trees = {}  # {model_id: model_root_node}
        self.model_params = {}  # {model_id: {param_id: [leaf_nodes]}} - save all block nodes for each parameter
        self.param_blocks_data = {}  # {model_id: {param_id: [block1, block2, ...]}} - store chunked data for each parameter
        self.param_blocks_count = {}  # {model_id: {param_id: num_blocks}} - store block count for each parameter
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
        Calculate secure hash value of data

        Args:
            data: Data to be hashed

        Returns:
            str: Hexadecimal hash value of data
        """
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, str):
            data_bytes = data.encode('UTF-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('UTF-8')

            # Initialize SHA-256 hash function
        sha256 = hashlib.sha256()

        # Update hash function
        sha256.update(data_bytes)

        # Return hexadecimal digest
        return sha256.hexdigest()

    def generate_secure_code(self, f=None):
        """
        Generate secure random code

        Args:
            f: Previous secure code (optional)

        Returns:
            str: Random integer string between 1 and 10000
        """
        # Generate random integer
        random_int = random.randint(1, 10000)

        # Convert to string and return
        return str(random_int)

    def encrypt_root_hash(self, root_hash):
        """
        Encrypt root hash using RSA

        Args:
            root_hash: Root hash value

        Returns:
            bytes: Encrypted hash value
        """
        if isinstance(root_hash, str):
            root_hash_bytes = root_hash.encode('UTF-8')
        elif isinstance(root_hash, bytes):
            root_hash_bytes = root_hash
        else:
            root_hash_bytes = str(root_hash).encode('UTF-8')

            # Encrypt using RSA public key
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
        Build Merkle Hash Tree

        Args:
            data_blocks: Data block list [(model_id, param_id, block_idx, data), ...]
            f: Secure code (optional)

        Returns:
            MHTNode: Root node of the tree
        """
        # Return None if no data blocks
        if not data_blocks:
            return None

            # If only one data block, directly create leaf node
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

            # Create all leaf nodes
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

            # Recursively build upper layer nodes
        return self._build_tree_from_nodes(leaf_nodes, f)

    def _build_tree_from_nodes(self, nodes, f=None):
        """
        Build tree from node list

        Args:
            nodes: Node list
            f: Secure code

        Returns:
            MHTNode: Root node
        """
        # Base case: return the node when there's only one node
        if len(nodes) == 1:
            return nodes[0]

            # Process nodes in pairs
        parent_nodes = []

        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # Check if there's a right node
            right_node = nodes[i + 1] if i + 1 < len(nodes) else None

            # Generate secure code
            if f is None:
                secure_code = self.generate_secure_code()
            else:
                secure_code = f

            # Create parent node
            parent = MHTNode()
            parent.left = left_node
            parent.right = right_node

            # Connect parent-child relationships
            left_node.parent = parent
            if right_node:
                right_node.parent = parent

                # Calculate combined hash value
            if right_node:
                combined_data = str(left_node.hash_value) + str(right_node.hash_value) + secure_code
            else:
                combined_data = str(left_node.hash_value) + secure_code

            parent.hash_value = self.secure_hash(combined_data)
            parent.secure_code = secure_code

            # Add to parent node list
            parent_nodes.append(parent)

            # Recursively build upper layer tree
        return self._build_tree_from_nodes(parent_nodes, secure_code)

    def chunk_parameters(self, params_array, chunk_size):
        """
        Split parameter array into fixed-size chunks

        Args:
            params_array: Parameter array
            chunk_size: Size of each chunk

        Returns:
            chunks: List of parameter chunks
            time_taken: Time taken for chunking operation
        """
        start_time = time.time()

        # Split parameter array
        if isinstance(params_array, np.ndarray):
            # Use NumPy array splitting operations
            # Ensure using flatten() to flatten the array
            flat_array = params_array.flatten()
            total_elements = len(flat_array)
            num_chunks = int(np.ceil(total_elements / chunk_size))

            chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_elements)
                chunks.append(flat_array[start_idx:end_idx])
        else:
            # Handle non-NumPy array cases
            chunks = [params_array[i:i + chunk_size] for i in range(0, len(params_array), chunk_size)]

        time_taken = time.time() - start_time

        print(f"Parameter chunking completed, generated {len(chunks)} chunks, took {time_taken:.4f} seconds")

        return chunks, time_taken

    def determine_chunk_size(self, params_array, target_chunks=16):
        """
        Determine optimal size for parameter chunking

        Args:
            params_array: Parameter array to be chunked
            target_chunks: Target number of chunks (default 16 chunks)

        Returns:
            chunk_size: Size of each chunk
            num_chunks: Actual number of chunks
        """
        if isinstance(params_array, np.ndarray):
            total_elements = params_array.size
        else:
            total_elements = len(params_array)

            # Calculate chunk size, round up to ensure all parameters are included
        chunk_size = max(1, int(np.ceil(total_elements / target_chunks)))

        # Calculate actual number of chunks
        num_chunks = int(np.ceil(total_elements / chunk_size))

        print(f"Total parameters: {total_elements}, target chunks: {target_chunks}")
        print(f"Calculated chunk size: {chunk_size}, actual chunks: {num_chunks}")

        return chunk_size, num_chunks

    def build_from_model_params(self, all_model_params: Dict[str, Dict[str, np.ndarray]], target_chunks=16):
        """
        Build Merkle Hash Tree from model parameters and encrypt root hash

        Args:
            all_model_params: Dictionary {model_id: {param_id: param_data}}
            target_chunks: Target number of chunks for each parameter

        Returns:
            dict: Performance statistics dictionary
        """
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        print(f"Building MHT, including {len(all_model_params)} models")

        # Initialize parameter block count and storage
        self.param_blocks_count = {}
        self.param_blocks_data = {}
        self.model_params = {}

        # Convert all model parameters to leaf data blocks
        all_data_blocks = []

        # First chunk each model parameter
        for model_id in all_model_params.keys():
            model_params = all_model_params[model_id]
            self.param_blocks_count[model_id] = {}
            self.param_blocks_data[model_id] = {}
            self.model_params[model_id] = {}

            for param_id in model_params.keys():
                param_data = model_params[param_id]
                self.param_blocks_data[model_id][param_id] = []

                if isinstance(param_data, np.ndarray):
                    # Flatten large arrays then chunk
                    flat_params = param_data.flatten()
                    chunk_size, num_chunks = self.determine_chunk_size(flat_params, target_chunks)
                    chunks, _ = self.chunk_parameters(flat_params, chunk_size)

                    # Save number of chunks
                    self.param_blocks_count[model_id][param_id] = len(chunks)

                    # Add each chunk to parameter block list and global data block list
                    for i, chunk in enumerate(chunks):
                        block_data = chunk.tobytes()
                        self.param_blocks_data[model_id][param_id].append(block_data)
                        all_data_blocks.append((model_id, param_id, i, block_data))
                else:
                    # Also chunk non-array types
                    data_str = str(param_data)
                    chunk_size, num_chunks = self.determine_chunk_size(data_str, target_chunks)
                    chunks = [data_str[i:i + chunk_size] for i in range(0, len(data_str), chunk_size)]

                    # Save number of chunks
                    self.param_blocks_count[model_id][param_id] = len(chunks)

                    # Add each chunk to parameter block list and global data block list
                    for i, chunk in enumerate(chunks):
                        block_data = chunk.encode('UTF-8')
                        self.param_blocks_data[model_id][param_id].append(block_data)
                        all_data_blocks.append((model_id, param_id, i, block_data))

        print(f"Total {len(all_data_blocks)} data blocks ready for tree building")

        # Use secure code to generate initial value
        initial_secure_code = self.generate_secure_code()

        # Build the entire tree
        self.root = self.build_mht(all_data_blocks, initial_secure_code)

        # Update model parameter node mapping - process after tree building
        self._update_model_params_mapping(self.root)

        # Record timestamp and version
        self.timestamp = int(time.time())
        self.version = 1

        # Get root hash
        root_hash = self.root.hash_value

        # Encrypt root hash
        encrypted_hash = self.encrypt_root_hash(root_hash)

        total_time = (time.time() - start_time)
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Calculate storage size
        tree_size = self._calculate_tree_size()

        print(f"\nTree building successful!")
        print(f"Root hash value: 0x{root_hash[:16] if isinstance(root_hash, str) else root_hash.hex()[:16]}...")
        print(f"Time taken: {total_time:.4f} seconds")
        print(f"Memory usage: {memory_used:.2f}MB")
        print(f"Tree size: {tree_size / (1024 * 1024):.2f}MB")

        # Record performance statistics
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
        Update mapping from model parameters to leaf nodes

        Args:
            node: Node to start searching from
        """
        if node is None:
            return

        if node.is_leaf:
            # Found leaf node, update mapping
            if node.model_id and node.param_id is not None:
                if node.model_id not in self.model_params:
                    self.model_params[node.model_id] = {}

                if node.param_id not in self.model_params[node.model_id]:
                    self.model_params[node.model_id][node.param_id] = []

                    # Add leaf node to parameter mapping
                self.model_params[node.model_id][node.param_id].append(node)
        else:
            # Recursively process child nodes
            self._update_model_params_mapping(node.left)
            self._update_model_params_mapping(node.right)

    def _calculate_tree_size(self):
        """Calculate total size of the tree (in bytes)"""
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

            # Calculate node size
            node_size = sys.getsizeof(node)

            # Add hash value size
            if node.hash_value:
                node_size += sys.getsizeof(node.hash_value)

                # Add secure code size
            if node.secure_code:
                node_size += sys.getsizeof(node.secure_code)

                # Add data size (if leaf node)
            if node.is_leaf and node.data:
                if isinstance(node.data, np.ndarray):
                    node_size += node.data.nbytes
                else:
                    node_size += sys.getsizeof(node.data)

            total_size += node_size

            # Add child nodes
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return total_size

    def save_to_file(self, output_file, include_public_key=True):
        """
        Save MHT metadata to file

        Args:
            output_file: Output file path
            include_public_key: Whether to include public key
        """
        if not self.root:
            print("Error: Tree is empty, cannot save")
            return

            # Encrypt root hash
        encrypted_hash = self.encrypt_root_hash(self.root.hash_value)

        # Create serializable tree representation
        tree_data = {
            'timestamp': self.timestamp,
            'version': self.version,
            'root_hash': self.root.hash_value if isinstance(self.root.hash_value, str) else self.root.hash_value.hex(),
            'encrypted_root_hash': encrypted_hash.hex(),
            'performance_stats': self.performance_stats,
            'param_blocks_count': self.param_blocks_count
        }

        # Add public key (if needed)
        if include_public_key:
            tree_data['public_key_pem'] = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')

        with open(output_file, 'w') as f:
            json.dump(tree_data, f, indent=2)

        print(f"MHT metadata saved to {output_file}")

    def get_model_proof(self, model_id_str: str) -> Dict[str, Any]:
        """
        Get proof path for a model
        """
        if model_id_str not in self.model_params:
            raise ValueError(f"Model {model_id_str} does not exist")

        model_params = self.model_params[model_id_str]
        blocks_data = self.param_blocks_data[model_id_str]

        # Build model parameter proof
        params_data = {}
        params_proofs = {}

        for param_id, leaf_nodes in model_params.items():
            # Sort leaf nodes by block index
            sorted_nodes = sorted(leaf_nodes, key=lambda x: x.block_idx)

            # Build proof path for each block
            blocks_proofs = []
            for node in sorted_nodes:
                # Get proof path from leaf node to root
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

                # Sort blocks and save parameter data
            params_data[param_id] = blocks_data[param_id]

            # Save all block proof paths for the parameter
            params_proofs[param_id] = blocks_proofs

        return {
            'model_id': model_id_str,
            'params': params_data,  # This is a dictionary with param_id as key and block list as value
            'params_proofs': params_proofs,
            'global_root_hash': self.root.hash_value,
            'timestamp': self.timestamp,
            'version': self.version,
            'param_blocks_count': self.param_blocks_count[model_id_str]
        }

    def verify_model_proof(self, proof: Dict[str, Any]) -> bool:
        """
        Verify model proof

        Args:
            proof: Proof obtained through get_model_proof

        Returns:
            bool: Whether verification was successful
        """
        # Extract proof information
        model_id = proof['model_id']
        global_root_hash = proof['global_root_hash']
        params = proof['params']  # {param_id: [block1, block2, ...]}
        params_proofs = proof['params_proofs']
        param_blocks_count = proof.get('param_blocks_count', {})

        # Verify each parameter
        for param_id, param_blocks in params.items():
            if param_id not in params_proofs:
                print(f"Proof for parameter {param_id} is missing")
                return False

            blocks_proofs = params_proofs[param_id]

            # Verify block count
            if param_id in param_blocks_count and len(blocks_proofs) != param_blocks_count[param_id]:
                print(f"Block count mismatch for parameter {param_id}: expected {param_blocks_count[param_id]}, actual {len(blocks_proofs)}")
                return False

                # Verify each block
            for proof_item in blocks_proofs:
                block_idx = proof_item['block_idx']

                if block_idx >= len(param_blocks):
                    print(f"Block index {block_idx} out of range")
                    return False

                block_data = param_blocks[block_idx]

                # Calculate leaf node hash
                leaf_hash = self.secure_hash(block_data)

                # Verify step by step according to proof path
                current_hash = leaf_hash
                for step in proof_item['proof']:
                    sibling_hash = step['hash']
                    secure_code = step.get('secure_code', '')

                    if step['position'] == 'left':
                        # Current node is on the right
                        combined = sibling_hash + current_hash + secure_code
                    else:
                        # Current node is on the left
                        combined = current_hash + sibling_hash + secure_code

                    current_hash = self.secure_hash(combined)

                    # Verify if we finally get the global root hash
                if current_hash != global_root_hash:
                    print(f"Proof verification failed for parameter {param_id} block {block_idx}")
                    return False

        return True

def save_merkle_hash_tree(model_tree, filepath):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        # Serialize object using pickle and save
        with open(filepath, 'wb') as f:
            pickle.dump(model_tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"ChameleonHashTree successfully saved to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving ChameleonHashTree: {e}")
        return False


def load_merkle_hash_tree(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File does not exist {filepath}")
        return None

    try:
        # Load object using pickle
        with open(filepath, 'rb') as f:
            model_tree = pickle.load(f)
        print(f"ChameleonHashTree successfully loaded from {filepath}")
        return model_tree
    except Exception as e:
        print(f"Error loading ChameleonHashTree: {e}")
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

        # Load encrypted model parameters
        encrypted_data = np.load(encrypted_path, allow_pickle=True)

        # Handle different types of NumPy arrays
        if isinstance(encrypted_data, np.ndarray) and encrypted_data.dtype == np.dtype('O'):
            # Handle object arrays
            if encrypted_data.ndim == 0:
                # 0-dimensional object array - use item() to get the dictionary inside
                model_params = encrypted_data.item()
                if not isinstance(model_params, dict):
                    print(f"Warning: Data for model {model_id} is not in dictionary format")
                    model_params = {"parameters": model_params}
            else:
                # Multi-dimensional object array - usually the first element of the array
                if len(encrypted_data) > 0 and isinstance(encrypted_data[0], dict):
                    model_params = encrypted_data[0]
                else:
                    print(f"Warning: Data format for model {model_id} is not the expected dictionary array")
                    model_params = {"full_array": encrypted_data}
        else:
            # Not an object array, might be a direct numerical array
            print(f"Data for model {model_id} is in simple array format")
            model_params = {"parameters": encrypted_data}

            # Add parameters to all model data
        print(f"Processing model {model_id}, parameter count: {len(model_params)}")
        for name, param in model_params.items():
            all_models_data[model_id][name] = param
            if isinstance(param, np.ndarray):
                print(f"  Parameter {name}: shape {param.shape}, type {param.dtype}")

    mht_builder = MerkleHashTree()
    # Increasing target_chunks can increase tree building overhead
    MHT, performance = mht_builder.build_from_model_params(all_models_data, target_chunks=18024)

    # Save MHT metadata
    mht_builder.save_to_file("mht_metadata.json")

    save_merkle_hash_tree(MHT, "./MHT_8.tree")

    return performance


if __name__ == "__main__":
    performance = get_MHT_multi_model()
    print(performance)
