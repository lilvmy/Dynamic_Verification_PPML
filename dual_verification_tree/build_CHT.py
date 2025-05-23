
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
    Storage for public key of chameleon hash
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
    Storage for private key of chameleon hash and relevant parameters
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
    Node of CHT
    """

    def __init__(self):
        self.hash_value = None  # hash value of node (bytes)
        self.rho = None  # random number rho
        self.delta = None  # random number delta
        self.left = None  # left sub-node
        self.right = None  # right sub-node
        self.parent = None  # parent node
        self.is_leaf = False  # whether it's a leaf node
        self.data = None  # reference to leaf node data

    def size_in_bytes(self):
        """Calculate memory usage of CHTNode (bytes)"""
        # Basic object overhead
        size = sys.getsizeof(self)

        # Calculate hash_value size
        if self.hash_value is not None:
            size += sys.getsizeof(self.hash_value)

        # Calculate rho size
        if self.rho is not None:
            size += sys.getsizeof(self.rho)

        # Calculate delta size
        if self.delta is not None:
            size += sys.getsizeof(self.delta)

        # Note: We don't calculate left, right and parent sizes
        # because these are references to other nodes, calculating them would lead to double counting
        # Only calculate the size of the references themselves
        size += sys.getsizeof(self.left)
        size += sys.getsizeof(self.right)
        size += sys.getsizeof(self.parent)

        # Calculate is_leaf size
        size += sys.getsizeof(self.is_leaf)

        # Calculate data size
        if self.data is not None:
            if isinstance(self.data, np.ndarray):
                size += self.data.nbytes
            elif hasattr(self.data, 'size_in_bytes') and callable(getattr(self.data, 'size_in_bytes')):
                # If data object has its own size_in_bytes method
                size += self.data.size_in_bytes()
            else:
                size += sys.getsizeof(self.data)

        return size

    # ====================== CHT ======================
class ChameleonHashTree:
    """CHT based on discrete log"""

    def __init__(self, keys: PrivateKeySet, security_param: int = 512):
        """
        Initialize CHT
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
        Calculate storage size of entire CHT tree (bytes)

        Args:
            include_values (bool): Whether to include data values (hash_value, rho, delta, data) in nodes in calculation

        Returns:
            int: Storage size (bytes)
        """
        if self.root is None:
            return 0

        # Use set to track visited nodes, avoid duplicate calculations
        visited = set()
        total_size = 0

        # Use stack for traversal, avoid stack overflow from recursion
        stack = [self.root]

        while stack:
            node = stack.pop()

            # Skip if node already visited
            if id(node) in visited:
                continue

            # Mark as visited
            visited.add(id(node))

            # Calculate current node size
            if include_values:
                # Calculate basic object size
                node_size = sys.getsizeof(node)

                # Calculate attribute sizes
                if node.hash_value is not None:
                    node_size += sys.getsizeof(node.hash_value)

                if node.rho is not None:
                    node_size += sys.getsizeof(node.rho)

                if node.delta is not None:
                    node_size += sys.getsizeof(node.delta)

                # is_leaf boolean size is already included in basic object size

                # Calculate data size
                if node.data is not None:
                    if isinstance(node.data, np.ndarray):
                        node_size += node.data.nbytes
                    elif hasattr(node.data, 'size_in_bytes') and callable(getattr(node.data, 'size_in_bytes')):
                        # If data has size_in_bytes method
                        node_size += node.data.size_in_bytes()
                    else:
                        node_size += sys.getsizeof(node.data)
            else:
                # Only calculate node structure overhead, not including values
                node_size = sys.getsizeof(node)

            # Add to total size
            total_size += node_size

            # Add child nodes to stack
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        return total_size

    def _generate_node_name(self, node_type="internal", model_id=None, param_id=None):
        """Generate unique node name for visualization"""
        self.node_count += 1
        if node_type == "leaf" and param_id:
            short_param = param_id.split('.')[-1][:5]  # Simplify parameter name
            return f"L{self.node_count}:{short_param}"
        elif node_type == "model" and model_id:
            short_model = model_id[:3]  # Simplify model ID
            return f"M{self.node_count}:{short_model}"
        else:
            return f"N{self.node_count}"

    def build_from_model_params(self, all_model_params: Dict[str, Dict[str, bytes]],signing_key: ecdsa.SigningKey) -> CHTNode:
        """
        Build CHT from model parameters
        """
        print(f"Building CHT, including {len(all_model_params)} models")

        # Build sub-tree for each model
        model_roots = []
        for model_id_str in all_model_params.keys():
            print(f"    Building model {model_id_str} sub tree:")
            model_params = all_model_params[model_id_str]

            # Generate leaf nodes for model
            leaf_nodes = []
            param_map = {}

            for param_id_str in model_params.keys():
                node = CHTNode()
                node.is_leaf = True
                node.data = model_params[param_id_str]
                # Store parameter ID for visualization
                node.param_id = param_id_str

                # Encode param info to compute hash
                encoded_data = self._encode_param(model_id_str, param_id_str, model_params[param_id_str])

                # Generate random numbers
                node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                # Compute hash value of leaf node
                node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                leaf_nodes.append(node)
                param_map[param_id_str] = node
                hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])
                print(f"  Parameter {param_id_str} leaf node: hash value = 0x{hash_str}...")

            model_root = self._build_internal_nodes(leaf_nodes)
            # Store model ID for visualization
            model_root.model_id = model_id_str[0:4]
            model_roots.append(model_root)

            # Save model sub tree and param mapping
            self.model_trees[model_id_str] = model_root
            self.model_params[model_id_str] = param_map

            hash_str = ''.join(f'{b:02x}' for b in model_root.hash_value[:4])
            print(f"  Model {model_id_str} sub-tree built successfully, root hash: 0x{hash_str}...")

        self.root = self._build_internal_nodes(model_roots)

        # Sign root node
        self.timestamp = int(time.time())
        self.version = 1
        root_hash_hex = ''.join(f'{b:02x}' for b in self.root.hash_value)
        message = f"{root_hash_hex}|{self.timestamp}|{self.version}".encode()
        self.signature = signing_key.sign(message, hashfunc=hashlib.sha256)

        print(
            f"Global tree built successfully, root hash: 0x{''.join(f'{b:02x}' for b in self.root.hash_value[:8])}...")

        return self.root

    def _encode_param(self, model_id: str, param_id: str, data: bytes) -> bytes:
        """
        Encode model param to compute hash
        """
        model_bytes = model_id.encode('utf-8')
        param_bytes = param_id.encode('utf-8')

        # Add length prefix to ensure unique decoding
        model_len = len(model_bytes).to_bytes(2, byteorder='big')
        param_len = len(param_bytes).to_bytes(2, byteorder='big')

        return model_len + model_bytes + param_len + param_bytes + data

    def _build_internal_nodes(self, nodes: List[CHTNode]) -> CHTNode:
        """
        Recursively build internal nodes
        """
        if len(nodes) == 1:
            return nodes[0]

        parent_nodes = []

        # Create parent nodes in pairs
        for i in range(0, len(nodes), 2):
            left_node = nodes[i]

            # If right node exists
            if i + 1 < len(nodes):
                right_node = nodes[i + 1]

                # Build parent node
                parent = CHTNode()
                parent.left = left_node
                parent.right = right_node
                left_node.parent = parent
                right_node.parent = parent

                # Hash value combining left and right nodes
                combined_data = left_node.hash_value + right_node.hash_value

                parent.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                parent.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                parent.hash_value = ChameleonHash.hash(combined_data, parent.rho, parent.delta, self.public_keys)

                parent_nodes.append(parent)
            else:
                # When there are an odd number of nodes, directly promote the last node
                parent_nodes.append(left_node)

        # Recursively build upper nodes
        return self._build_internal_nodes(parent_nodes)

    def get_model_proof(self, model_id_str: str) -> Dict[str, Any]:
        """
        Get proof path for a model
        """
        if model_id_str not in self.model_trees:
            raise ValueError(f"Model {model_id_str} does not exist")

        model_root = self.model_trees[model_id_str]
        model_params = self.model_params[model_id_str]

        # Build proof path from sub tree root to global root
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

        # Build verification path for model
        params_data = {}
        params_proofs = {}

        for param_id_str, leaf_node in model_params.items():
            # Get param
            params_data[param_id_str] = leaf_node.data

            # Get proof path from leaf node to sub tree root
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

            # Save proof path of param
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
        """Update models or parameters (add model, delete model, modify parameters) while maintaining CHT integrity

        Args:
            model_to_add: New model to add {model_id: {param_id: data, ...}}
            model_id_to_delete: Model ID to delete
            param_modifications: Parameters to modify {model_id: {param_id: new_data, ...}}

        Returns:
            Updated tree root node
        """
        # Track modified paths
        modified_paths = []

        # 1. Handle deleting entire model
        if model_id_to_delete:
            if model_id_to_delete not in self.model_trees:
                print(f"Model {model_id_to_delete} does not exist, cannot delete")
            else:
                try:
                    # Get the model subtree root node to delete
                    model_root = self.model_trees[model_id_to_delete]

                    # We cannot actually delete nodes from the tree, but can replace it with an empty model
                    # Create an empty model containing a single empty parameter
                    empty_param_id = "_empty_"
                    empty_param_data = b''

                    # Create leaf node for empty model
                    node = CHTNode()
                    node.is_leaf = True
                    node.data = empty_param_data
                    node.param_id = empty_param_id  # Add parameter ID for visualization

                    # Encode parameter information
                    encoded_data = self._encode_param(model_id_to_delete, empty_param_id, empty_param_data)

                    # Generate random numbers
                    node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                    node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                    # Calculate leaf node hash
                    node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                    # This single node now represents the entire deleted model
                    new_model_root = node
                    new_model_root.model_id = model_id_to_delete[0:4]  # Add model ID for visualization

                    # Update path from model root to global root
                    current_node = model_root
                    new_node = new_model_root

                    # Save modified path information
                    path = []

                    # Recursively update parent nodes upward until reaching global root
                    while current_node.parent is not None:
                        parent = current_node.parent

                        # Record modified path
                        parent_hash_before = parent.hash_value

                        # Determine if current node is left or right node
                        is_left = current_node == parent.left

                        # Get sibling node
                        sibling = parent.right if is_left else parent.left

                        # Create new encoded data (concatenation of left and right child hashes)
                        if is_left:
                            combined_data = new_node.hash_value + sibling.hash_value
                        else:
                            combined_data = sibling.hash_value + new_node.hash_value

                        # Find hash collision for new data
                        pre_image = ChameleonHash.forge(parent.hash_value, combined_data, self.keys)

                        # If left node, update parent's left child, otherwise update right child
                        if is_left:
                            parent.left = new_node
                        else:
                            parent.right = new_node

                        new_node.parent = parent

                        # Update parent's rho and delta
                        parent.rho = pre_image.rho
                        parent.delta = pre_image.delta

                        # Verify hash value remains unchanged
                        new_hash = ChameleonHash.hash(combined_data, pre_image.rho, pre_image.delta, self.public_keys)
                        if new_hash != parent.hash_value:
                            raise Exception("Hash collision verification failed when updating tree structure")

                        # Record node modification before and after
                        path.append({
                            'node_level': 'internal',
                            'original_hash': parent_hash_before.hex(),
                            'new_hash': parent.hash_value.hex(),
                            'position': 'left' if is_left else 'right',
                            'rho': pre_image.rho,
                            'delta': pre_image.delta
                        })

                        # Move to next level for continued processing
                        current_node = parent
                        new_node = parent

                    # Update model mapping
                    # Create new empty parameter mapping
                    empty_params = {empty_param_id: node}

                    # Update model tree and parameter mapping
                    self.model_trees[model_id_to_delete] = new_model_root
                    self.model_params[model_id_to_delete] = empty_params

                    print(f"Model {model_id_to_delete} successfully marked for deletion, maintaining global tree structure")

                except Exception as e:
                    print(f"Failed to delete model {model_id_to_delete}: {str(e)}")

        # 2. Handle modifying model parameters
        if param_modifications:
            # First update leaf node data
            modified_model_ids = set()
            for model_id, params in param_modifications.items():
                if model_id not in self.model_params:
                    for param_id in params:
                        print(f"Parameter modification failed: Model {model_id} does not exist")
                    continue

                model_params = self.model_params[model_id]
                model_modified = False

                for param_id, new_data in params.items():
                    if param_id not in model_params:
                        print(f"Parameter modification failed: Parameter {param_id} does not exist in model {model_id}")
                        continue

                    try:
                        # Update leaf node data
                        leaf_node = model_params[param_id]
                        leaf_node.data = new_data
                        model_modified = True
                        print(f"Parameter (model {model_id} param {param_id}) data updated, preparing to rebuild tree")

                    except Exception as e:
                        print(f"Failed to modify parameter (model {model_id} param {param_id}): {str(e)}")

                # Record if model was modified
                if model_modified:
                    modified_model_ids.add(model_id)

            # Start rebuilding tree
            print(f"Starting tree structure rebuild, involving {len(modified_model_ids)} modified models...")

            try:
                # Rebuild all modified model subtrees
                for model_id in modified_model_ids:
                    model_params_dict = {}
                    # Collect all parameters of the model
                    for param_id, leaf_node in self.model_params[model_id].items():
                        model_params_dict[param_id] = leaf_node.data

                    # Clear old model subtree
                    old_model_root = self.model_trees[model_id]

                    # Create new leaf nodes and subtree for modified model
                    leaf_nodes = []
                    param_map = {}

                    for param_id_str in model_params_dict.keys():
                        node = CHTNode()
                        node.is_leaf = True
                        node.data = model_params_dict[param_id_str]
                        node.param_id = param_id_str

                        # Encode parameter data
                        encoded_data = self._encode_param(model_id, param_id_str, model_params_dict[param_id_str])

                        # Generate new hash parameters
                        node.rho = ChameleonHash.get_random_in_range(self.keys.get_q())
                        node.delta = ChameleonHash.get_random_in_range(self.keys.get_q())

                        # Calculate new hash value
                        node.hash_value = ChameleonHash.hash(encoded_data, node.rho, node.delta, self.public_keys)

                        leaf_nodes.append(node)
                        param_map[param_id_str] = node
                        hash_str = ''.join(f'{b:02x}' for b in node.hash_value[:4])
                        print(f"  Rebuilt parameter {param_id_str} leaf node: hash value = 0x{hash_str}...")

                    # Build model subtree
                    new_model_root = self._build_internal_nodes(leaf_nodes)
                    new_model_root.model_id = model_id[0:8]

                    # Update model tree and parameter mapping
                    self.model_trees[model_id] = new_model_root
                    self.model_params[model_id] = param_map

                    hash_str = ''.join(f'{b:02x}' for b in new_model_root.hash_value[:4])
                    print(f"  Model {model_id} subtree rebuilt successfully, root hash: 0x{hash_str}...")

                # Collect all model subtree root nodes
                all_model_roots = list(self.model_trees.values())

                # Save current global root hash
                old_root_hash = self.root.hash_value.hex() if self.root else None

                # Rebuild global tree
                new_global_root = self._build_internal_nodes(all_model_roots)

                # Update global root node
                self.root = new_global_root

                # Record new root hash
                new_root_hash = self.root.hash_value.hex()

                print(f"Global root hash changed from {old_root_hash[:8]}... to {new_root_hash[:8]}...")

            except Exception as e:
                print(f"Failed to rebuild tree structure: {str(e)}")
                traceback.print_exc()

            print(
                f"Total modified {sum(len(params) for model_id, params in param_modifications.items())} parameters")

            # Return updated tree
            return self

        # 3. Handle adding entire model
        if model_to_add:
            try:
                # Build subtree for new model
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
                        print(f"  Parameter {param_id_str} leaf node: hash value = 0x{hash_str}...")

                    new_model_root = self._build_internal_nodes(leaf_nodes)

                    new_model_root.model_id = model_id_str[0:4]
                    new_model_roots.append(new_model_root)

                    self.model_trees[model_id_str] = new_model_root
                    self.model_params[model_id_str] = param_map

                    hash_str = ''.join(f'{b:02x}' for b in new_model_root.hash_value[:4])
                    print(f"  Model {model_id_str} sub-tree built successfully, root hash: 0x{hash_str}...")

                self.root = self._build_internal_nodes(new_model_roots)

                # First, get current all model subtree root nodes
                current_model_roots = list(self.model_trees.values())

                # Merge current model roots and new model roots
                all_model_roots = current_model_roots

                # Save current global root hash
                old_root_hash = self.root.hash_value.hex() if self.root else None

                # Rebuild global tree
                new_global_root = self._build_internal_nodes(all_model_roots)

                # Update global root node
                self.root = new_global_root

                # Record new root hash
                new_root_hash = self.root.hash_value.hex()

            except Exception as e:
                print(f"Failed to add model: {str(e)}")

        # Return tree structure
        return self

def draw_tree(tree_or_root, output_file="./figure/CHT.png", max_depth=15):
    """
    Draw chameleon hash tree using optimized layout algorithm, adaptive node spacing, and simplified node labels

    Args:
        tree_or_root: Can be ChameleonHashTree object or CHTNode object
        output_file: Output file path
        max_depth: Maximum display depth, set larger value to show complete tree
    """
    # Set matplotlib parameters to ensure font embedding and high quality output
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
    mpl.rcParams['ps.fonttype'] = 42  # Use TrueType fonts
    mpl.rcParams['svg.fonttype'] = 'none'  # Use SVG native text
    mpl.rcParams['figure.dpi'] = 150  # Default DPI

    # Use standard sans-serif fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

    # Determine root node
    if isinstance(tree_or_root, ChameleonHashTree):
        root = tree_or_root.root
        model_trees = tree_or_root.model_trees if hasattr(tree_or_root, 'model_trees') else {}
        model_params = tree_or_root.model_params if hasattr(tree_or_root, 'model_params') else {}
    else:
        root = tree_or_root
        model_trees = {}
        model_params = {}

    if root is None:
        print("Error: Root node is empty, cannot draw tree")
        return

    # Create reverse mapping: from node to model ID and parameter ID
    node_to_model_id = {}
    node_to_param_id = {}

    # Fill mapping information
    for model_id, model_root in model_trees.items():
        node_to_model_id[id(model_root)] = model_id

        if model_id in model_params:
            for param_id, param_node in model_params[model_id].items():
                node_to_param_id[id(param_node)] = param_id

    # Collect tree levels and leaf nodes

    def analyze_tree(node, level=0):
        if node is None:
            return [], {}, [], 0

        # Process current node
        all_nodes = [(node, level)]
        level_map = {id(node): level}
        max_lvl = level

        # Collect leaf nodes
        leaf_nodes = []
        if hasattr(node, 'is_leaf') and node.is_leaf:
            leaf_nodes.append(node)

        # Process left subtree
        if hasattr(node, 'left') and node.left:
            left_nodes, left_level_map, left_leaves, left_max = analyze_tree(node.left, level + 1)
            all_nodes.extend(left_nodes)
            level_map.update(left_level_map)
            leaf_nodes.extend(left_leaves)
            max_lvl = max(max_lvl, left_max)

        # Process right subtree
        if hasattr(node, 'right') and node.right:
            right_nodes, right_level_map, right_leaves, right_max = analyze_tree(node.right, level + 1)
            all_nodes.extend(right_nodes)
            level_map.update(right_level_map)
            leaf_nodes.extend(right_leaves)
            max_lvl = max(max_lvl, right_max)

        return all_nodes, level_map, leaf_nodes, max_lvl

    # Analyze tree structure

    all_nodes, level_map, leaf_nodes, max_level = analyze_tree(root)

    # Ensure leaf nodes are all at bottom level
    for node in leaf_nodes:
        level_map[id(node)] = max_level

    # Calculate number of nodes per level
    level_counts = {}
    for _, level in level_map.items():
        level_counts[level] = level_counts.get(level, 0) + 1

    print(f"Tree depth: {max_level}, leaf nodes: {len(leaf_nodes)}")
    print(f"Level distribution: {level_counts}")

    # Create graph for visualization
    G = nx.DiGraph()

    # Convert parameter ID to hexadecimal prefix
    def get_hex_prefix(param_id):
        """Get first four hex digits from parameter ID"""
        try:
            # Try using hash function to get hash value of parameter ID
            import hashlib
            hash_obj = hashlib.md5(param_id.encode('utf-8'))
            return hash_obj.hexdigest()[:4]
        except:
            # If failed, simply return first 4 characters of parameter ID
            return param_id[:4] if len(param_id) >= 4 else param_id

    # BFS traverse tree, create graph structure

    def build_graph(node, parent_id=None):
        if node is None:
            return

        # Create unique node ID
        node_id = id(node)
        str_node_id = str(node_id)

        # Determine node label and attributes - use simplified labels
        if node == root:
            label = "Root"  # Simplified Global Root -> Root
            color = "red"
            shape = "o"  # Square
            size = 3000
            group = "root"
        elif node_id in node_to_model_id:
            # Model root node - only keep cnnx part
            model_id = node_to_model_id[node_id]
            # Extract model name, without "Model:" prefix
            if ":" in model_id:
                parts = model_id.split(":")
                label = parts[-1]
            else:
                label = model_id
            color = "orange"
            shape = "o"  # Circle
            size = 2500
            group = "model"
        elif hasattr(node, 'is_leaf') and node.is_leaf:
            # Leaf node - only keep first four hex digits of parameter ID
            if node_id in node_to_param_id:
                param_id = node_to_param_id[node_id]
                # Get hex prefix of parameter ID
                label = get_hex_prefix(param_id)
            else:
                # When no parameter ID, use last four digits of node ID
                label = str_node_id[-4:]
            color = "lightgreen"
            shape = "o"  # Triangle
            size = 2000
            group = "param"
        else:
            # Internal node - only keep trailing numbers
            node_num = str_node_id.split('-')[-1] if '-' in str_node_id else str_node_id[-4:]
            label = node_num
            color = "skyblue"
            shape = "o"  # Circle
            size = 2000
            group = "internal"

        # Add node
        G.add_node(str_node_id,
                   label=label,
                   color=color,
                   level=level_map[node_id],
                   shape=shape,
                   size=size,
                   group=group,
                   is_leaf=hasattr(node, 'is_leaf') and node.is_leaf)

        # Add edge
        if parent_id:
            G.add_edge(parent_id, str_node_id)

        # Process child nodes
        if hasattr(node, 'left') and node.left:
            build_graph(node.left, str_node_id)
        if hasattr(node, 'right') and node.right:
            build_graph(node.right, str_node_id)

    # Build graph

    build_graph(root)

    # Check if graph is empty
    if len(G.nodes()) == 0:
        print("Error: Generated graph has no nodes, please check tree structure")
        return

    print(f"Successfully created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

    # ======================== Layout Algorithm ========================

    # Subtree size of each node (calculated by leaf node count)
    subtree_sizes = {}

    def count_subtree_leaves(node_id):
        """Calculate number of leaf nodes in subtree rooted at node_id"""
        children = list(G.successors(node_id))

        # If leaf node
        if not children:
            if G.nodes[node_id].get('is_leaf', False):
                subtree_sizes[node_id] = 1
                return 1
            else:
                subtree_sizes[node_id] = 0
                return 0

        # If internal node, calculate sum of leaf counts from all child nodes
        size = sum(count_subtree_leaves(child) for child in children)
        subtree_sizes[node_id] = size
        return size

    # Calculate subtree size for each node

    for node in G.nodes():
        if not list(G.predecessors(node)):  # Find root node
            count_subtree_leaves(node)

    # Horizontal position allocation

    def assign_x_positions(node_id, start_pos, available_width):
        """Allocate horizontal positions for nodes, based on relative space allocation by subtree size"""
        children = list(G.successors(node_id))
        node_level = G.nodes[node_id]['level']

        # Set current node position
        if node_id not in pos:
            pos[node_id] = (start_pos + available_width / 2, -node_level * 5)

        # If no children, return
        if not children:
            return

        # Allocate child node positions
        total_subtree_size = sum(subtree_sizes[child] for child in children)
        # Minimum unit width, ensure small subtrees have enough space
        min_unit_width = available_width / (len(children) * 2)

        current_pos = start_pos
        for child in children:
            # Calculate child subtree space proportion, considering minimum width
            if total_subtree_size > 0:
                child_width = max(
                    min_unit_width,
                    (subtree_sizes[child] / total_subtree_size) * available_width
                )
            else:
                child_width = available_width / len(children)

            # Recursively allocate child node positions
            assign_x_positions(child, current_pos, child_width)
            current_pos += child_width

    # Initialize position dictionary

    pos = {}

    # Find root node
    root_node_id = None
    for node in G.nodes():
        if not list(G.predecessors(node)):
            root_node_id = node
            break

    # Estimate required total width - based on leaf node count and node count per level
    total_leaf_count = len([n for n in G.nodes() if G.nodes[n].get('is_leaf', False)])
    max_nodes_per_level = max(level_counts.values())

    # Use adaptive width factor - consider leaf node count and maximum nodes per level
    width_factor = max(total_leaf_count * 6, max_nodes_per_level * 15)

    # Allocate positions - use width factor to determine total width
    assign_x_positions(root_node_id, 0, width_factor)

    # ======================== Draw Tree ========================

    # Calculate image size
    max_x = max(x for x, _ in pos.values())
    min_x = min(x for x, _ in pos.values())
    max_y = abs(min(y for _, y in pos.values()))

    # Calculate appropriate image size - decide based on node count and distribution
    fig_width = max(20, (max_x - min_x) / 100 + 5)  # Add margin
    fig_height = max(10, max_y / 30 + 3)

    plt.figure(figsize=(fig_width, fig_height), dpi=150)
    plt.clf()  # Clear current figure

    # Draw edges - use arcs to reduce crossings
    for u, v in G.edges():
        # Calculate edge curvature - based on horizontal distance
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = abs(x1 - x2)

        # Dynamically adjust curvature based on horizontal distance
        if dx > 20:
            rad = min(0.3, dx / 200)  # Greater distance, greater curvature, but with upper limit
        else:
            rad = 0.05  # Small curvature for small distances

        # Draw edge
        nx.draw_networkx_edges(G, pos,
                               edgelist=[(u, v)],
                               arrows=True,
                               edge_color="gray",
                               width=1.0,
                               connectionstyle=f'arc3,rad={rad}',
                               arrowstyle="-|>",
                               arrowsize=12,
                               alpha=0.7)

    # Group and draw nodes by type
    node_groups = {
        'root': [n for n in G.nodes() if G.nodes[n]['group'] == 'root'],
        'model': [n for n in G.nodes() if G.nodes[n]['group'] == 'model'],
        'internal': [n for n in G.nodes() if G.nodes[n]['group'] == 'internal'],
        'param': [n for n in G.nodes() if G.nodes[n]['group'] == 'param']
    }

    # Shapes and sizes for different node groups
    shapes = {'root': 'o', 'model': 'o', 'internal': 'o', 'param': 'o'}
    sizes = {'root': 3000, 'model': 2500, 'internal': 2000, 'param': 2000}
    colors = {'root': 'pink', 'model': 'orange', 'internal': 'skyblue', 'param': 'lightgreen'}

    # Draw each group of nodes
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

    # Different font sizes for different node groups
    font_sizes = {'root': 13, 'model': 12, 'internal': 10, 'param': 9}

    # Draw labels
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

    # Add legend
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

    # Set figure parameters
    plt.axis('off')  # Don't show coordinate axes
    plt.tight_layout(pad=0.3)  # Compact layout but leave space

    # Save based on file type
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Determine file format
    file_ext = os.path.splitext(output_file)[1].lower()

    if file_ext == '.svg':
        # SVG format - best text embedding option
        plt.savefig(output_file, format='svg', bbox_inches='tight')
        print(f"Tree diagram saved as SVG format with fully embedded text: {output_file}")
    elif file_ext == '.pdf':
        # PDF format - good text embedding
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"Tree diagram saved as PDF format with embedded text: {output_file}")
    else:
        # PNG or other bitmap format - use high DPI
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Tree diagram saved as bitmap format using high DPI (300): {output_file}")

    # Display image
    plt.show()

    return G  # Return graph object for further analysis


def save_chameleon_hash_tree(model_tree, filepath):
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


def load_chameleon_hash_tree(filepath):
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

def main():
    # Load cht_keys_params
    key_path = "../key_storage/cht_keys_params.key"
    cht_keys = load_cht_keys(key_path)

    # Load ecdsa keys
    ecdsa_private_key, ecdsa_public_key = load_ecdsa_keys()

    all_models_data = {}
    model_id_mapping = {}
    # Get model id
    with open("/home/lilvmy/paper-demo/Results_Verification_PPML/model_id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    print(model_id_mapping)
    # Get encrypted model params
    for model_id, encrypted_path in model_id_mapping.items():
        all_models_data[model_id] = {}
        encrypted_model_param = extract_data_from_hash_node(encrypted_path)
        print(encrypted_model_param)
        for name, param in encrypted_model_param.items():
            all_models_data[model_id][name] = param

    # Build model verification tree CHT
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
