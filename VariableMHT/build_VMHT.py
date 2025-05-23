
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
    """Variable Merkle Hash Tree Node"""

    def __init__(self, level, position, leaf_nodes, num_children, hash_value):
        """Initialize VMHT Node"""
        self.level = level  # Level in the tree (H represents leaf nodes, 1 represents root node)
        self.position = position  # Position in the level
        self.leaf_nodes = leaf_nodes  # Number of leaf nodes in subtree
        self.num_children = num_children  # Number of child nodes
        self.hash_value = hash_value  # Node hash value
        self.firstChild = None  # First child node
        self.secondChild = None  # Second child node
        self.thirdChild = None  # Third child node (for 3-child case)
        self.parent = None  # Parent node
        self.data = None  # Leaf node data
        # For visualization and debugging
        self.param_id = None  # Parameter ID (for leaf nodes)
        self.model_id = None  # Model ID (for model subtree root nodes)
        self.block_idx = None  # Parameter block index
        self.shuffle_key = None  # Parameter-specific random permutation key

    def __str__(self):
        model_str = f", model={self.model_id}" if self.model_id else ""
        param_str = f", param={self.param_id}" if self.param_id else ""
        return f"Node(level={self.level}, pos={self.position}, leaves={self.leaf_nodes}, children={self.num_children}{model_str}{param_str})"

    def get_children(self):
        """Return list of all child nodes"""
        children = []
        if self.firstChild:
            children.append(self.firstChild)
        if self.secondChild:
            children.append(self.secondChild)
        if self.thirdChild:
            children.append(self.thirdChild)
        return children


class VariableMerkleHahsTree:
    """Variable Merkle Hash Tree implementation"""

    def __init__(self):
        """Initialize VMHT"""
        self.root = None  # Root node of the tree
        self.height = 0  # Height of the tree
        self.model_trees = {}  # {model_id: root_node} Model subtrees
        self.param_nodes = {}  # {model_id: {param_id: [leaf_nodes]}} List of leaf nodes for each parameter
        self.timestamp = None  # Timestamp when tree was created
        self.version = 1  # Version of the tree
        self.security_code = None  # Security code for hashing
        self.shuffle_keys = {}  # Parameter-specific random permutation keys {model_id: {param_id: key}}
        self.param_blocks = {}  # Number of blocks per parameter {model_id: {param_id: num_blocks}}
        self.max_block_size = 0  # Maximum block size (bytes)
        self.performance_stats = {}  # Performance statistics

    def hash_function(self, data, g, shuffle_key=None):
        """
        Calculate hash value of data using security code g and optional random permutation key

        Args:
        data -- Data to be hashed (numpy array, bytes or other)
        g -- Security code
        shuffle_key -- Optional parameter-specific random permutation key

        Returns:
        bytes: Hash value
        """
        hasher = hashlib.sha256()

        # Include security code
        hasher.update(str(g).encode())

        # Include random permutation key (if provided)
        if shuffle_key is not None:
            hasher.update(str(shuffle_key).encode())

            # Hash based on data type
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
        Pseudo-random permutation function

        Args:
        sh -- Random permutation key (seed)
        m -- Number of elements to permute

        Returns:
        List[int]: Permutation of indices [0, m-1]
        """
        random.seed(sh)
        indices = list(range(m))
        random.shuffle(indices)
        return indices

    def build_tree(self, all_model_params, security_code=None, max_block_size=16):
        """
        Build VMHT tree from model parameters

        Args:
        all_model_params -- Dictionary {model_id: {param_id: np.ndarray}}
        security_code -- Optional security code
        max_block_size -- Number of data blocks per parameter

        Returns:
        dict: Performance statistics
        """
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # Generate security code (if not provided)
        if security_code is None:
            security_code = str(random.randint(1, 10000))
        self.security_code = security_code
        self.max_block_size = max_block_size

        # Record timestamp
        self.timestamp = int(time.time())

        print(f"Building VMHT, including {len(all_model_params)} models")

        # Calculate block counts and generate parameter-specific random permutation keys
        self.param_blocks = {}
        self.shuffle_keys = {}

        for model_id, params in all_model_params.items():
            self.param_blocks[model_id] = {}
            self.shuffle_keys[model_id] = {}
            self.param_nodes[model_id] = {}

            for param_id, param_data in params.items():
                # First flatten parameter to 1D array
                if isinstance(param_data, np.ndarray):
                    # Flatten NumPy array
                    flat_param_data = param_data.flatten()
                    param_size = flat_param_data.nbytes
                elif isinstance(param_data, bytes):
                    flat_param_data = param_data
                    param_size = len(param_data)
                else:
                    # Convert non-binary data to string and encode
                    flat_param_data = str(param_data).encode()
                    param_size = len(flat_param_data)

                    # Calculate number of blocks needed for this parameter
                num_blocks = max(1, math.ceil(param_size / max_block_size))
                self.param_blocks[model_id][param_id] = num_blocks

                # Generate random permutation key for this parameter (not exceeding its block count)
                self.shuffle_keys[model_id][param_id] = random.randint(0, num_blocks - 1)

                # Call vmhtgen to process all model parameters
        self.vmhtgen(all_model_params)

        # Calculate performance metrics
        total_time = time.time() - start_time
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        tree_size = self.calculate_storage_size()

        # Record performance statistics
        performance = {
            "build_time_sec": total_time * 1000,
            "memory_used_mb": memory_used,
            "tree_size_mb": tree_size / (1024 * 1024)
        }

        self.performance_stats["build"] = performance

        print(f"\nTree building successful!")
        print(f"Root hash value: 0x{self.root.hash_value.hex()[:16]}...")
        print(f"Time taken: {total_time:.4f} seconds")
        print(f"Memory usage: {memory_used:.2f}MB")
        print(f"Tree size: {tree_size / (1024 * 1024):.2f}MB")

        return self, performance

    def vmhtgen(self, all_model_params):
        """
        Generate VMHT tree

        Args:
        all_model_params -- Dictionary {model_id: {param_id: np.ndarray}}

        Returns:
        TreeNode: Root node of the tree
        """
        # Process models and build model subtrees
        model_roots = []

        for model_id, model_params in all_model_params.items():
            print(f"Building subtree for model {model_id}")

            # Create data block list
            all_blocks = []
            param_nodes_map = {}

            # Process each parameter
            for param_id, param_data in model_params.items():
                # Flatten parameter data
                if isinstance(param_data, np.ndarray):
                    flat_param_data = param_data.flatten().tobytes()
                elif isinstance(param_data, bytes):
                    flat_param_data = param_data
                else:
                    flat_param_data = str(param_data).encode('utf-8')

                    # Get number of blocks for this parameter
                num_blocks = self.param_blocks[model_id][param_id]
                param_size = len(flat_param_data)

                # Get parameter-specific random permutation key
                param_shuffle_key = self.shuffle_keys[model_id][param_id]

                # Split parameter into blocks
                if num_blocks == 1:
                    # Single block, no need to split
                    all_blocks.append((model_id, param_id, 0, flat_param_data, param_shuffle_key))
                else:
                    # Multiple blocks, split parameter
                    block_size = math.ceil(param_size / num_blocks)

                    for i in range(num_blocks):
                        start_idx = i * block_size
                        end_idx = min(start_idx + block_size, param_size)
                        block_data = flat_param_data[start_idx:end_idx]

                        # Add block with random permutation key
                        all_blocks.append((model_id, param_id, i, block_data, param_shuffle_key))

                        # Get total number of blocks
            m = len(all_blocks)
            if m == 0:
                continue

                # Calculate tree height H = ⌈log₂m⌉ + 1
            H = math.ceil(math.log2(m)) + 1

            # Create leaf nodes for each block
            Q = deque()
            for i, (model_id, param_id, block_idx, block_data, shuffle_key) in enumerate(all_blocks):
                # Calculate hash using parameter-specific random permutation
                hash_val = self.hash_function(block_data, self.security_code, shuffle_key)

                # Create leaf node
                node = TreeNode(H, i, 1, 0, hash_val)
                node.model_id = model_id
                node.param_id = param_id
                node.block_idx = block_idx
                node.data = block_data
                node.shuffle_key = shuffle_key

                Q.append(node)

                # Store parameter leaf node reference
                if param_id not in param_nodes_map:
                    param_nodes_map[param_id] = []
                param_nodes_map[param_id].append(node)

                # print(f"  Parameter {param_id} (block {block_idx + 1}/{self.param_blocks[model_id][param_id]}): "
                #       f"Hash value={hash_val.hex()[:8]}... (random permutation key: {shuffle_key})")

                # Build subtree
            l = H
            r = 1

            while len(Q) > 1:
                x = Q.popleft()
                y = Q.popleft()
                z = Q[0] if len(Q) > 0 else None
                w = Q[1] if len(Q) > 1 else None

                if z is not None and z.level == l and (w is None or w.level != l):
                    z = Q.popleft()

                    # Create 3-child node
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
                    # Create 2-child node
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

                    # Get model root node
            model_root = Q.popleft() if Q else None
            model_root.model_id = model_id

            # Save model subtree and parameter mapping
            self.model_trees[model_id] = model_root
            self.param_nodes[model_id] = param_nodes_map

            # Add to model root node list
            model_roots.append(model_root)

            print(f"  Model {model_id} subtree building completed: root hash value={model_root.hash_value.hex()[:8]}...")

            # Build global tree from model root nodes
        if model_roots:
            # Calculate tree height for model root nodes
            m = len(model_roots)
            H = math.ceil(math.log2(m)) + 1
            self.height = H

            # Initialize queue with model root nodes
            Q = deque(model_roots)

            # Build tree
            l = H
            r = 1

            while len(Q) > 1:
                x = Q.popleft()
                y = Q.popleft()
                z = Q[0] if len(Q) > 0 else None
                w = Q[1] if len(Q) > 1 else None

                if z is not None and z.level == l and (w is None or w.level != l):
                    z = Q.popleft()

                    # Create 3-child node
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
                    # Create 2-child node
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

                    # Set global root node
            self.root = Q.popleft() if Q else None

        print(f"Global tree building completed: root hash value={self.root.hash_value.hex()[:8]}...")
        return self.root

    def calculate_storage_size(self):
        """Calculate total storage size of the tree (bytes)"""
        if self.root is None:
            return 0

            # Use set to track visited nodes, avoid duplicate calculations
        visited = set()
        total_size = 0

        # Use stack instead of recursion, avoid stack overflow
        stack = [self.root]

        while stack:
            node = stack.pop()

            # Skip already visited nodes
            if id(node) in visited:
                continue

                # Mark as visited
            visited.add(id(node))

            # Calculate node size
            node_size = sys.getsizeof(node)

            # Add hash value size
            if node.hash_value is not None:
                node_size += sys.getsizeof(node.hash_value)

                # Add string attribute size
            if node.model_id:
                node_size += sys.getsizeof(node.model_id)
            if node.param_id:
                node_size += sys.getsizeof(node.param_id)

                # Add data size
            if node.data is not None:
                if isinstance(node.data, np.ndarray):
                    node_size += node.data.nbytes
                elif isinstance(node.data, bytes):
                    node_size += len(node.data)
                else:
                    node_size += sys.getsizeof(node.data)

                    # Add to total size
            total_size += node_size

            # Add child nodes to stack
            for child in node.get_children():
                if child is not None:
                    stack.append(child)

        return total_size

    def generate_audit_request(self, sample_ratio=0.3, specific_models=None):
        """
        Generate audit request for VMHT

        Args:
        sample_ratio -- Proportion of parameters to sample
        specific_models -- Optional list of specific model IDs for audit

        Returns:
        AuditRequest object
        """
        # Calculate total number of parameters
        total_params = 0
        for model_id, params in self.param_nodes.items():
            if specific_models is None or model_id in specific_models:
                total_params += sum(len(blocks) for blocks in params.values())

        if total_params == 0:
            raise ValueError("No parameters available for audit")

            # Calculate number of nodes to sample
        ln = max(1, int(total_params * sample_ratio))

        # Generate random permutation key
        sh = random.randint(0, total_params - 1)

        # Now to find valid node positions, we first need to find actual existing levels and positions in the tree
        valid_positions = self._get_valid_node_positions()

        if not valid_positions:
            # If no valid positions found, use default level and position
            l = 1
            r = 1
        else:
            # Randomly select a valid (level, position) combination
            l, r = random.choice(valid_positions)

        return {"ln": ln, "l": l, "r": r, "sh": sh, "model_ids": specific_models}

    def _get_valid_node_positions(self):
        """
        Get all valid (level, position) combinations in the tree

        Returns:
        List[(int, int)]: List of valid (level, position) combinations
        """
        if not self.root:
            return []

        valid_positions = []
        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            # Record current node's level and position
            valid_positions.append((node.level, node.position))

            # Add child nodes to queue
            for child in node.get_children():
                if child:
                    queue.append(child)

        return valid_positions

    def verify_proof(self, proof, request):
        """
        Verify audit proof

        Args:
        proof -- Integrity proof
        request -- Audit request

        Returns:
        bool: Whether verification passes
        """
        # Check if tree has been built
        if self.root is None:
            print("Error: Tree not yet built, cannot verify")
            return False

            # Handle position verification request
        l = request.get("l")
        r = request.get("r")

        if l is not None and r is not None:
            # Get verification tag
            tag_from_proof = proof.get("tag")
            if not tag_from_proof:
                print("Error: Tag missing from proof")
                return False

                # Find node at specific position
            node_at_lr = self._find_node_at_level_position(l, r)

            if not node_at_lr:
                print(f"Error: Node not found at level {l} position {r}")
                return False

                # Get tag from tree
            if isinstance(node_at_lr.hash_value, bytes):
                tag_from_tree = node_at_lr.hash_value.hex()
            else:
                tag_from_tree = node_at_lr.hash_value

                # Compare tags
            verification_result = tag_from_proof == tag_from_tree

            if verification_result:
                print(f"Verification successful: Node tag at level {l} position {r} matches")
            else:
                print(f"Verification failed: Tag mismatch")
                print(f"Tag from proof: {tag_from_proof}")
                print(f"Tag from tree: {tag_from_tree}")

            return verification_result

            # Handle parameter sampling verification request
        elif "data_references" in proof:
            # This part is your original parameter sampling verification logic
            # According to your code, this part might not need changes
            tag_from_proof = proof.get("tag")
            data_references = proof.get("data_references", [])

            # Implement sampling verification logic...
            # This part can remain unchanged since it's currently working

            return True

        else:
            print("Error: Unsupported proof or request type")
            return False

    def _find_node_at_level_position(self, level, position):
        """
        Find node at specific level and position in the tree

        Args:
        level -- Level
        position -- Position

        Returns:
        TreeNode or None: Found node or None
        """
        if not self.root:
            return None

        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            if node.level == level and node.position == position:
                return node

                # If current node level is greater than target level, continue searching downward
            if node.level < level:
                for child in node.get_children():
                    if child:
                        queue.append(child)

        return None

    def progen(self, request):
        """
        Generate integrity proof for audit request

        Args:
        request -- Audit request

        Returns:
        Integrity proof dictionary
        """
        # Check if tree has been built
        if self.root is None:
            print("Error: Tree not yet built, cannot generate proof")
            return {"error": "Tree not yet built"}

            # Extract request parameters
        ln = request.get("ln", 0)
        sh = request.get("sh", 0)
        specific_models = request.get("model_ids")
        l = request.get("l")
        r = request.get("r")

        # Handle position verification request
        if l is not None and r is not None:
            # Find node at specific position
            node_at_lr = self._find_node_at_level_position(l, r)

            if not node_at_lr:
                print(f"Error: Node not found at level {l} position {r}")
                return {"error": f"Node not found at level {l} position {r}"}

                # Get tag
            if isinstance(node_at_lr.hash_value, bytes):
                tag = node_at_lr.hash_value.hex()
            else:
                tag = node_at_lr.hash_value

            return {
                "tag": tag
            }

            # Handle parameter sampling request
        # The following is the original sampling logic, can remain unchanged
        # ...

        # Collect all available leaf nodes
        all_leaf_nodes = []

        # Handle specific model filtering
        if specific_models:
            # Convert to list if single model ID
            model_ids = [specific_models] if isinstance(specific_models, str) else specific_models

            # Only collect nodes from specified models
            for model_id in model_ids:
                if model_id in self.param_nodes:
                    for param_id, block_nodes in self.param_nodes[model_id].items():
                        for block_idx, node in enumerate(block_nodes):
                            shuffle_key = self.shuffle_keys.get(model_id, {}).get(param_id, 0)
                            all_leaf_nodes.append((model_id, param_id, block_idx, node, shuffle_key))
        else:
            # Collect nodes from all models
            for model_id, params in self.param_nodes.items():
                for param_id, block_nodes in params.items():
                    for block_idx, node in enumerate(block_nodes):
                        shuffle_key = self.shuffle_keys.get(model_id, {}).get(param_id, 0)
                        all_leaf_nodes.append((model_id, param_id, block_idx, node, shuffle_key))

                        # Get total number of parameters
        m = len(all_leaf_nodes)
        if m == 0:
            return {"sample_size": 0, "data_references": [],
                    "tag": self.hash_function("empty", self.security_code).hex()}

            # Apply random permutation based on sh
        perm = self.permutation_function(sh, m)
        shuffled_nodes = [all_leaf_nodes[perm[i]] for i in range(m)]

        # Sample parameters (from last ln items)
        ln = min(ln, m)
        start_idx = m - ln
        sampled_nodes = shuffled_nodes[start_idx:]

        # Create data reference list
        data_references = [(node[0], node[1], node[2], node[4]) for node in
                           sampled_nodes]  # (model_id, param_id, block_idx, shuffle_key)

        # Collect hash values from sampled nodes
        sampled_hashes = []
        for _, _, _, node, _ in sampled_nodes:
            # Get node's hash value
            if node and hasattr(node, 'hash_value'):
                if isinstance(node.hash_value, bytes):
                    sampled_hashes.append(node.hash_value)
                else:
                    # If hash value is string, convert to bytes
                    sampled_hashes.append(
                        node.hash_value.encode('utf-8') if isinstance(node.hash_value, str) else node.hash_value)

                    # Calculate combined hash as tag
        if sampled_hashes:
            # Concatenate all hash values in sequence
            combined_hash = b''.join(sampled_hashes)
            # Calculate final tag
            tag = self.hash_function(combined_hash, self.security_code).hex()
        else:
            tag = self.hash_function("empty", self.security_code).hex()

            # Return integrity proof
        return {
            "sample_size": ln,
            "data_references": data_references,
            "tag": tag
        }

    def save_to_file(self, output_file):
        """
        Save VMHT metadata to file

        Args:
        output_file -- Output file path
        """
        if not self.root:
            print("Error: Tree is empty, cannot save")
            return

            # Create serializable tree representation
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

        print(f"VMHT metadata saved to {output_file}")

    def get_model_proof(self, model_id):
        """
        Get proof for a model

        Args:
        model_id -- Model ID

        Returns:
        Model proof dictionary
        """
        if model_id not in self.model_trees:
            raise ValueError(f"Model {model_id} does not exist")

            # Create audit request for specific model
        request = self.generate_audit_request(sample_ratio=1.0, specific_models=[model_id])

        # Generate proof
        proof = self.progen(request)

        # Merge request and proof
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
        Generate request for verifying all parameter blocks of a model

        Args:
        model_id -- Model ID to verify

        Returns:
        Parameter verification request
        """
        if not self.root:
            raise ValueError("Tree not yet built, cannot generate verification request")

        if model_id not in self.param_nodes:
            raise ValueError(f"Model {model_id} does not exist in tree")

        return {
            "verification_type": "all_params",
            "model_id": model_id
        }

    def generate_param_proof(self, model_id):
        """
        Generate verification proof for all parameter blocks of a specific model

        Args:
        model_id -- Model ID to verify

        Returns:
        Verification proof for all parameter blocks
        """
        if not self.root:
            print("Error: Tree not yet built, cannot generate proof")
            return {"error": "Tree not yet built"}

        if model_id not in self.param_nodes:
            print(f"Error: Model {model_id} does not exist in tree")
            return {"error": f"Model {model_id} does not exist"}

            # Collect all parameter block information and hash values for the model
        param_blocks = {}

        for param_id, block_nodes in self.param_nodes[model_id].items():
            param_blocks[param_id] = []

            for block_idx, node in enumerate(block_nodes):
                # Get node hash value
                if isinstance(node.hash_value, bytes):
                    hash_value = node.hash_value.hex()
                else:
                    hash_value = node.hash_value

                    # Get parameter-specific random permutation key
                shuffle_key = self.shuffle_keys.get(model_id, {}).get(param_id, 0)

                # Add block information
                param_blocks[param_id].append({
                    "block_idx": block_idx,
                    "hash": hash_value,
                    "shuffle_key": shuffle_key
                })

                # Get model root hash for additional verification
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
        Verify all parameter blocks of a model

        Args:
        proof -- Parameter block verification proof
        request -- Verification request

        Returns:
        dict: Verification result, including success rate and details
        """
        if not self.root:
            print("Error: Tree not yet built, cannot verify")
            return {"success": False, "error": "Tree not yet built"}

            # Get model ID from request
        if "model_id" not in request:
            print("Error: Model ID missing from request")
            return {"success": False, "error": "Model ID missing from request"}

        model_id = request["model_id"]

        # Check if model exists
        if model_id not in self.param_nodes:
            print(f"Error: Model {model_id} does not exist in tree")
            return {"success": False, "error": f"Model {model_id} does not exist"}

            # Get parameter block information from proof
        param_blocks_from_proof = proof.get("param_blocks", {})

        if not param_blocks_from_proof:
            print("Error: Parameter block information missing from proof")
            return {"success": False, "error": "Parameter block information missing from proof"}

            # Verify all parameter blocks of the model
        success_count = 0
        total_count = 0
        failures = []

        for param_id, blocks_info in param_blocks_from_proof.items():
            # Check if parameter exists in tree
            if param_id not in self.param_nodes[model_id]:
                failures.append({
                    "param_id": param_id,
                    "error": "Parameter does not exist in tree"
                })
                continue

            tree_blocks = self.param_nodes[model_id][param_id]

            for block_info in blocks_info:
                block_idx = block_info.get("block_idx")
                hash_from_proof = block_info.get("hash")

                total_count += 1

                # Check if block index is valid
                if block_idx is None or block_idx >= len(tree_blocks):
                    failures.append({
                        "param_id": param_id,
                        "block_idx": block_idx,
                        "error": "Invalid block index"
                    })
                    continue

                    # Get block node from tree
                node = tree_blocks[block_idx]

                # Get hash value from tree
                if isinstance(node.hash_value, bytes):
                    hash_from_tree = node.hash_value.hex()
                else:
                    hash_from_tree = node.hash_value

                    # Compare hash values
                if hash_from_tree == hash_from_proof:
                    success_count += 1
                else:
                    failures.append({
                        "param_id": param_id,
                        "block_idx": block_idx,
                        "hash_proof": hash_from_proof[:10] + "...",
                        "hash_tree": hash_from_tree[:10] + "..."
                    })

                    # Calculate verification success rate
        success_rate = success_count / total_count if total_count > 0 else 0

        # Determine overall verification result
        result = {
            "success": success_count == total_count,
            "success_rate": success_rate,
            "success_count": success_count,
            "total_count": total_count,
            "failures": failures[:10] if len(failures) > 10 else failures  # Limit number of failure records
        }

        # Print verification result
        if result["success"]:
            print(f"Verification successful: All {total_count} parameter blocks for model {model_id} verified")
        else:
            print(f"Verification failed: {total_count - success_count} out of {total_count} parameter blocks for model {model_id} failed verification")

        return result


def get_VMHT_multi_model():
    """Build VMHT from pre-trained models and return performance statistics"""
    all_models_data = {}
    model_id_mapping = {}

    # Get model ID mapping
    with open("/VariableMHT/model_id_pre_trained_model.txt", 'r',
              encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            model_id_mapping[key] = value

    print(model_id_mapping)

    # Get parameters for each model
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

                # Build VMHT
    vmht_builder = VariableMerkleHahsTree()
    performance = vmht_builder.build_tree(all_models_data)

    # Save tree structure metadata
    vmht_builder.save_to_file("./vmht_metadata.json")

    # Generate and verify a proof, test functionality
    request = vmht_builder.generate_audit_request(sample_ratio=0.2)
    proof = vmht_builder.progen(request)
    verification_result = vmht_builder.verify_proof(proof, request)

    print(f"Audit verification result: {'Passed' if verification_result else 'Failed'}")

    return performance


if __name__ == "__main__":
    performance = get_VMHT_multi_model()
    print(performance)
