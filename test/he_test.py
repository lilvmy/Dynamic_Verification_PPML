from level_homomorphic_encryption.HE import CKKSPyfhel


n_mults = 6
m = 8192
scale_power = 25

encryption_parameters = {
    'm': m,                      # For CKKS, n/2 values can be encoded in a single ciphertext
    'scale': 2**scale_power,                 # Each multiplication grows the final scale
    'qi': [34]+ [scale_power]*n_mults +[34]  # One intermdiate for each multiplication
}

HE_Client = CKKSPyfhel(**encryption_parameters)
HE_Client.generate_keys()
HE_Client.generate_relin_keys()

public_key = HE_Client.get_public_key()
relin_key  = HE_Client.get_relin_key()


HE_Server = CKKSPyfhel(**encryption_parameters)
print(HE_Server.load_public_key(public_key))
HE_Server.load_relin_key(relin_key)
