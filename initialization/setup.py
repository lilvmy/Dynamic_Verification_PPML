from level_homomorphic_encryption.HE import CKKSPyfhel
from typing import List, Dict, Tuple, Any, Optional
import ecdsa
import pickle

def setup_he():
    """
    store HE's private key, public key, relin key
    """
    n_mults = 6
    m = 8192
    scale_power = 25

    encryption_parameters = {
        'm': m,  # For CKKS, n/2 values can be encoded in a single ciphertext
        'scale': 2 ** scale_power,  # Each multiplication grows the final scale
        'qi': [34] + [scale_power] * n_mults + [34]  # One intermdiate for each multiplication
    }

    HE= CKKSPyfhel(**encryption_parameters)
    HE.generate_keys()
    HE.generate_relin_keys()

    secret_key = HE.get_secret_key()
    rotate_key = HE.get_rotate_key()
    public_key = HE.get_public_key()
    relin_key = HE.get_relin_key()

    return secret_key, rotate_key, public_key, relin_key


def load_HE_keys():
    """
    load HE's public key, secret key, relin key
    """
    n_mults = 6
    m = 8192
    scale_power = 25

    encryption_parameters = {
        'm': m,  # For CKKS, n/2 values can be encoded in a single ciphertext
        'scale': 2 ** scale_power,  # Each multiplication grows the final scale
        'qi': [34] + [scale_power] * n_mults + [34]  # One intermdiate for each multiplication
    }

    HE= CKKSPyfhel(**encryption_parameters)

    with open("../key_storage/pub.key", "rb") as f:
        public_key = f.read()
    HE.load_public_key(public_key)

    with open("../key_storage/secret.key", "rb") as f:
        secret_key = f.read()
    HE.load_secret_key(secret_key)

    with open("../key_storage/relin.key", "rb") as f:
        relin_key = f.read()
    HE.load_relin_key(relin_key)

    with open("../key_storage/rotate.key", "rb") as f:
        rotate_key = f.read()
    HE.load_rotate_key(rotate_key)

    return HE

def save_ecdsa_keys() -> Tuple[ecdsa.SigningKey, ecdsa.VerifyingKey]:
    """
    store signature's private key, public key
    use NIST P-256 curve (secp256r1)
    """
    sk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
    vk = sk.verifying_key
    with open("../key_storage/ecdsa_private.key", "wb") as f:
        pickle.dump(sk, f)

    with open("../key_storage/ecdsa_public.key", "wb") as f:
        pickle.dump(vk, f)


    return sk, vk

def load_ecdsa_keys():
    """
    load ecdsa private key, public key
    """
    with open("../key_storage/ecdsa_private.key", "rb") as f:
        ecdsa_private_key = pickle.load(f)

    with open("../key_storage/ecdsa_public.key", "rb") as f:
        ecdsa_public_key = pickle.load(f)

    return ecdsa_private_key, ecdsa_public_key


    





if __name__ == "__main__":

    # secret_key, rotate_key, public_key, relin_key = setup_he()
    # print(f"the secret key is {secret_key}")
    # print(f"the rotate key is {rotate_key}")
    # print(f"the public key is {public_key}")
    # print(f"the relin key is {relin_key}")

    # HE = load_HE_keys()
    # plaintext1 = [1.23, 2, 3]
    # plaintext2 = [2.23, 3, 4]
    # ciphertext1 = HE.encrypt_matrix(plaintext1)
    # print(f"the ciphertext is {ciphertext1}")
    # ciphertext2 = HE.encode_matrix(plaintext2)
    # cipher = ciphertext1 + ciphertext2
    # decrypted_results1 = HE.decrypt_matrix(cipher)
    # print(f"the decrypted results is {decrypted_results1}")
    # # save_ecdsa_keys()

    # sk, vk = load_ecdsa_keys()
    # print(sk)
    # print(vk)

    setup_he()








