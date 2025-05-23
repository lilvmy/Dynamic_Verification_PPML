import hashlib
import random
import secrets
import pickle

class PublicKeySet:
    """
    storage public key of chameleon hash
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
    storage private key of chameleon hash and relevant parameters
    """

    def __init__(self, p, q, g, sk, pk):
        super().__init__(p, q, g, pk)
        self.sk = sk  # private key

    def get_secret_key(self):
        return self.sk

    def get_public_key_set(self):
        return PublicKeySet(self.p, self.q, self.g, self.pk)


class PreImage:
    """
    storage message and pre-image of the random number
    """

    def __init__(self, data, rho, delta):
        self.data = data
        self.rho = rho
        self.delta = delta


class ChameleonHash:
    """
    realize chameleon hash based on discrete log
    """

    CERTAINTY = 80

    @staticmethod
    def get_safe_prime(t: int) -> int:
        """generate a security prime p, p=2q+1
        """
        while True:
            # generate random prime q
            q = secrets.randbits(t - 1)  # the bytes of q is t
            if q % 2 == 0:
                q += 1

                # judge q is prime based on Miller-Rabin experiments
            if ChameleonHash.is_probable_prime(q, ChameleonHash.CERTAINTY):
                #
                p = 2 * q + 1
                # experiments p is prime
                if ChameleonHash.is_probable_prime(p, ChameleonHash.CERTAINTY):
                    return p

    @staticmethod
    def is_probable_prime(n: int, k: int) -> bool:
        """
        realize Miller-Rabin experiments
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False

            # define n - 1 as d*2^r
        r, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            r += 1

            # k round Miller-Rabin experiments
        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue

            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False

        return True

    @staticmethod
    def get_random_in_range(n: int) -> int:
        """
        get random number from [1, n-1]
        """
        return random.randint(1, n - 1)

    @staticmethod
    def find_generator(p: int) -> int:
        """
        find generator of module p
        for p = 2q+1, find generator of sub-group with order q
        """
        while True:
            h = ChameleonHash.get_random_in_range(p)
            g = pow(h, 2, p)
            if g != 1:
                return pow(g, 2, p)

    @staticmethod
    def crypto_hash(a: int, b: int, bit_length: int) -> int:
        """
        crypto hash function with variable len
        """
        # join a and b (bytes)
        x = (a.to_bytes((a.bit_length() + 7) // 8, 'big') +
             b.to_bytes((b.bit_length() + 7) // 8, 'big'))

        # initialize SHA3
        md = hashlib.sha3_256()
        md.update(x)
        hash_value = int.from_bytes(md.digest(), 'big')

        # if the len of hash is insufficient, continue hash and join
        while hash_value.bit_length() < bit_length:
            x = (int.from_bytes(x, 'big') + 1).to_bytes((len(x) + 1), 'big')
            md = hashlib.sha3_256()
            md.update(x)
            block = int.from_bytes(md.digest(), 'big')
            hash_value = (hash_value << block.bit_length()) | block

            # adjust to target length
        shift_back = hash_value.bit_length() - bit_length
        return hash_value >> shift_back

    @staticmethod
    def key_gen(t: int) -> PrivateKeySet:
        """
        generate public/private key
        """
        p = ChameleonHash.get_safe_prime(t)
        q = (p - 1) // 2

        g = ChameleonHash.find_generator(p)
        sk = ChameleonHash.get_random_in_range(q)
        pk = pow(g, sk, p)

        return PrivateKeySet(p, q, g, sk, pk)

    @staticmethod
    def hash(data: bytes, rho: int, delta: int, keys: PublicKeySet) -> bytes:
        """
        compute chameleon hash of message
        """
        e = ChameleonHash.crypto_hash(int.from_bytes(data, 'big'), rho, keys.get_p().bit_length())

        t1 = pow(keys.get_public_key(), e, keys.get_p())
        t2 = pow(keys.get_g(), delta, keys.get_p())
        ch = (rho - (t1 * t2) % keys.get_p()) % keys.get_q()

        return ch.to_bytes((ch.bit_length() + 7) // 8, 'big')

    @staticmethod
    def forge(hash_value: bytes, data: bytes, keys: PrivateKeySet) -> PreImage:
        """
        fake pre-image for new message
        """
        c = int.from_bytes(hash_value, 'big')
        m_prime = int.from_bytes(data, 'big')
        k = ChameleonHash.get_random_in_range(keys.get_q())

        rho_prime = (c + pow(keys.get_g(), k, keys.get_p()) % keys.get_q()) % keys.get_q()
        e_prime = ChameleonHash.crypto_hash(m_prime, rho_prime, keys.get_p().bit_length())
        delta_prime = (k - (e_prime * keys.get_secret_key())) % keys.get_q()

        return PreImage(data, rho_prime, delta_prime)


def load_cht_keys(key_path):
    with open(key_path, "rb") as f:
        cht_keys_dict = pickle.load(f)

    # extract parameters form cht_keys_dict and build PrivateKeySet
    return PrivateKeySet(
        p=cht_keys_dict['p'],
        q=cht_keys_dict['q'],
        g=cht_keys_dict['g'],
        sk=cht_keys_dict['secret_key'],
        pk=cht_keys_dict['public_key']
    )



def main():
    # generate keys of chameleon hash
    # save_keys_params = {}
    # ch_keys = ChameleonHash.key_gen(256)
    # public_keys_param = {}
    # public_keys_param["secret_key"] = ch_keys.get_secret_key()
    # public_keys_param["public_key"] = ch_keys.get_public_key()
    # public_keys_param["p"] = ch_keys.get_p()
    # public_keys_param["q"] = ch_keys.get_q()
    # public_keys_param["g"] = ch_keys.get_g()
    #
    # for name, value in public_keys_param.items():
    #     save_keys_params[name] = value
    # with open("../key_storage/cht_keys_params.key", "wb") as f:
    #     pickle.dump(save_keys_params, f)
    # print(f"generate chameleon hash keys successfully (p = {ch_keys.get_p().bit_length()} bits)")


    cht_keys1 = load_cht_keys("../key_storage/cht_keys_params.key")
    print(cht_keys1.get_public_key_set().get_q())

if __name__ == "__main__":
    main()
