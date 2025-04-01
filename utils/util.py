import numpy as np
import sys


def generate_random_matrices(count, shape_range=None, fixed_shape=None,
                             min_val=-10, max_val=10,
                             ensure_mixed_signs=True, seed=None):
    """
    生成多个包含正负元素的随机矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    matrices = []

    for i in range(count):
        # 确定当前矩阵的形状
        if fixed_shape is not None:
            rows, cols = fixed_shape
        elif shape_range is not None:
            (min_rows, max_rows), (min_cols, max_cols) = shape_range
            rows = np.random.randint(min_rows, max_rows + 1)
            cols = np.random.randint(min_cols, max_cols + 1)
        else:
            # 默认形状范围
            rows = np.random.randint(2, 11)
            cols = np.random.randint(2, 11)

        # 生成随机矩阵
        matrix = np.random.uniform(min_val, max_val, (rows, cols))

        # 确保矩阵同时包含正负元素
        if ensure_mixed_signs:
            while not (np.any(matrix > 0) and np.any(matrix < 0)):
                matrix = np.random.uniform(min_val, max_val, (rows, cols))

                # 如果经过多次尝试仍无法满足条件，强制添加正负元素
                if not np.any(matrix > 0):
                    matrix.flat[np.random.randint(0, matrix.size)] = np.random.uniform(0.1, max_val)
                if not np.any(matrix < 0):
                    matrix.flat[np.random.randint(0, matrix.size)] = np.random.uniform(min_val, -0.1)

        matrices.append(matrix)

    # 如果所有矩阵形状相同，返回3D数组
    if fixed_shape is not None or all(m.shape == matrices[0].shape for m in matrices):
        return np.stack(matrices)  # 使用np.stack代替np.array确保创建正确的维度
    else:
        # 形状不同时，返回包含矩阵的对象数组
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

