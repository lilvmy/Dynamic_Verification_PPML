import numpy as np
from FMIV.build_MHT import load_merkle_hash_tree, MerkleHashTree
import csv
import time
from utils.util import get_size


def client_get_normal_model():
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

        # 加载加密模型参数
        encrypted_data = np.load(encrypted_path, allow_pickle=True)

        # 处理不同类型的NumPy数组
        if isinstance(encrypted_data, np.ndarray) and encrypted_data.dtype == np.dtype('O'):
            # 处理对象数组
            if encrypted_data.ndim == 0:
                # 0维对象数组 - 使用item()获取其中的字典
                model_params = encrypted_data.item()
                if not isinstance(model_params, dict):
                    print(f"警告: 模型 {model_id} 的数据不是字典格式")
                    model_params = {"parameters": model_params}
            else:
                # 多维对象数组 - 通常是数组的第一个元素
                if len(encrypted_data) > 0 and isinstance(encrypted_data[0], dict):
                    model_params = encrypted_data[0]
                else:
                    print(f"警告: 模型 {model_id} 的数据格式不是预期的字典数组")
                    model_params = {"full_array": encrypted_data}
        else:
            # 不是对象数组，可能是直接的数值数组
            print(f"模型 {model_id} 的数据是简单数组格式")
            model_params = {"parameters": encrypted_data}

            # 将参数添加到所有模型数据中
        print(f"处理模型 {model_id}, 参数数量: {len(model_params)}")
        for name, param in model_params.items():
            all_models_data[model_id][name] = param
            if isinstance(param, np.ndarray):
                print(f"  参数 {name}: 形状 {param.shape}, 类型 {param.dtype}")

    mht_builder = MerkleHashTree()
    # 增加traget_chunks可以增加树的构建开销
    MHT, performance = mht_builder.build_from_model_params(all_models_data, target_chunks=18024)



    with open(f"./client_verification_time_storage_costs_untamper.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'time_costs', 'storage_costs'])

    for model_id_str in model_id_mapping.keys():
        start_time = time.time()
        client_request_model = model_id_str
        print(f"the client request {client_request_model} model")

        model_package = MHT.get_model_proof(client_request_model)
        model_package_size = get_size(model_package) / (1024 * 1024)

        print("\nthe client starts run verification operation:")
        results = MHT.verify_model_proof(model_package)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        with open(f"./client_verification_time_storage_costs_untamper.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_id_str, total_time, model_package_size])

        print(f"record experiment data: model name={model_id_str}, total time={total_time}ms, model verification package size={model_package_size}MB")


    return all_models_data

if __name__ == "__main__":
    model_datas = client_get_normal_model()


