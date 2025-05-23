import numpy as np
import sys


def generate_random_matrices(count, shape_range=None, fixed_shape=None,
                             min_val=-10, max_val=10,
                             ensure_mixed_signs=True, seed=None):
    """
    Generate multiple random matrices containing both positive and negative elements
    """
    if seed is not None:
        np.random.seed(seed)

    matrices = []

    for i in range(count):
        # Determine the shape of current matrix
        if fixed_shape is not None:
            rows, cols = fixed_shape
        elif shape_range is not None:
            (min_rows, max_rows), (min_cols, max_cols) = shape_range
            rows = np.random.randint(min_rows, max_rows + 1)
            cols = np.random.randint(min_cols, max_cols + 1)
        else:
            # Default shape range
            rows = np.random.randint(2, 11)
            cols = np.random.randint(2, 11)

        # Generate random matrix
        matrix = np.random.uniform(min_val, max_val, (rows, cols))

        # Ensure matrix contains both positive and negative elements
        if ensure_mixed_signs:
            while not (np.any(matrix > 0) and np.any(matrix < 0)):
                matrix = np.random.uniform(min_val, max_val, (rows, cols))

                # If condition cannot be met after multiple attempts, force adding positive and negative elements
                if not np.any(matrix > 0):
                    matrix.flat[np.random.randint(0, matrix.size)] = np.random.uniform(0.1, max_val)
                if not np.any(matrix < 0):
                    matrix.flat[np.random.randint(0, matrix.size)] = np.random.uniform(min_val, -0.1)

        matrices.append(matrix)

    # If all matrices have same shape, return 3D array
    if fixed_shape is not None or all(m.shape == matrices[0].shape for m in matrices):
        return np.stack(matrices)  # Use np.stack instead of np.array to ensure correct dimensions
    else:
        # When shapes are different, return object array containing matrices
        result = np.empty(count, dtype=object)
        for i, matrix in enumerate(matrices):
            result[i] = matrix
        return result

def get_size(obj, seen=None):
    """Recursively find the size of objects in memory"""
    if seen is None:
        seen = set()

    # If object already counted, return 0
    obj_id = id(obj)
    if obj_id in seen:
        return 0

        # Add this object to seen
    seen.add(obj_id)
    size = sys.getsizeof(obj)

    # Check for nested containers
    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        # For custom objects with attributes
        size += get_size(obj.__dict__, seen)

    return size

if __name__ == "__main__":
    res = generate_random_matrices(count=1, fixed_shape=(1, 64), min_val=-1, max_val=1)
    print(res)
