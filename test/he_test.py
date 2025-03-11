from level_homomorphic_encryption.HE import CKKSPyfhel


n_mults = 6
m = 2**13
scale_power = 25
ckks_parameters = {
    'm': m,
    'scale': 2**scale_power,
    'qi': [34] + [scale_power]*n_mults + [34]
}
HE = CKKSPyfhel(**ckks_parameters)
HE.generate_keys()
HE.generate_relin_keys()

a = HE.encrypt_number(5.50)
b = HE.encrypt_number(4.50)

c = HE.power(a, 2)
d = HE.decrypt(c)
print(d)