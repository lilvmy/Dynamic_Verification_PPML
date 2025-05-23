import numpy as np
from Pyfhel import Pyfhel
import tenseal as ts

import tempfile

tmp_dir = tempfile.TemporaryDirectory()




class HE:
    def generate_keys(self):
        pass

    def generate_relin_keys(self):
        pass

    def get_public_key(self):
        pass

    def get_relin_key(self):
        pass

    def get_secret_key(self):
        pass

    def get_rotate_key(self):
        pass

    def load_public_key(self, key):
        pass

    def load_relin_key(self, key):
        pass
    def load_secret_key(self, key):
        pass
    def load_rotate_key(self, key):
        pass

    def encode_matrix(self, matrix):
        """Encode a matrix in a plaintext HE nD-matrix.

        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded

        Returns
        -------
        matrix
            nD-np.array with encoded values
        """
        pass

    def decode_matrix(self, matrix):
        pass

    def encrypt_matrix(self, matrix):
        pass

    def decrypt_matrix(self, matrix):
        pass

    def encode_number(self, number):
        pass

    def power(self, number, exp):
        pass

    def noise_budget(self, ciphertext):
        pass


class BFVPyfhel(HE):
    def __init__(self, m=2048, p=None, p_bits=20, sec=128):
        self.he = Pyfhel()
        if p_bits is None:
            self.he.contextGen(scheme='bfv', t=p, n=m, sec=sec)
        else:
            self.he.contextGen(scheme='bfv', t_bits=p_bits, n=m, sec=sec)

    def encode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encodeInt(np.array([x], dtype=np.int64))

    def decode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decodeInt(x)[0]

    def encrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encryptInt(np.array([x], dtype=np.int64))

    def decrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decryptInt(x)[0]

    def generate_keys(self):
        self.he.keyGen()

    def generate_relin_keys(self):
        self.he.relinKeyGen()

    def get_public_key(self):
        self.he.save_public_key(tmp_dir.name + "/pub.key")
        with open(tmp_dir.name + "/pub.key", 'rb') as f:
            return f.read()

    def get_relin_key(self):
        self.he.save_relin_key(tmp_dir.name + "/relin.key")
        with open(tmp_dir.name + "/relin.key", 'rb') as f:
            return f.read()

    def load_public_key(self, key):
        with open(tmp_dir.name + "/pub.key", 'wb') as f:
            f.write(key)
        self.he.load_public_key(tmp_dir.name + "/pub.key")

    def load_relin_key(self, key):
        with open(tmp_dir.name + "/relin.key", 'wb') as f:
            f.write(key)
        self.he.load_relin_key(tmp_dir.name + "/relin.key")

    def encode_matrix(self, matrix):
        """Encode a float nD-matrix in a PyPtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded
        Returns
        -------
        matrix
            nD-np.array( dtype=PyPtxt ) with encoded values
        """

        try:
            return np.array(list(map(self.encode, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
        """Decode a PyPtxt nD-matrix in a float nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyPtxt )
            matrix to be decoded
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with float values
        """
        try:
            return np.array(list(map(self.decode, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
        """Encrypt a float nD-matrix in a PyCtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=PyCtxt ) with encrypted values
        """
        try:
            return np.array(list(map(self.encrypt, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])

    def decrypt_matrix(self, matrix):
        """Decrypt a PyCtxt nD matrix in a float nD matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyCtxt )
            matrix to be decrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with plain values
        """
        try:
            return np.array(list(map(self.decrypt, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])

    def encode_number(self, number):
        return self.encode(number)

    def power(self, number, exp):
        return self.he.power(number, exp)

    def noise_budget(self, ciphertext):
        try:
            return self.he.noise_level(ciphertext)
        except SystemError:
            return "Can't get NB without secret key."


class CKKSPyfhel(HE):
    def __init__(self, m=2**14, scale=2**30, qi=[60, 30, 30, 30, 60]):
        self.he = Pyfhel()
        self.he.contextGen(scheme='ckks', n=m, scale=scale, qi_sizes=qi)

    def encode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encodeFrac(np.array([x], dtype=np.float64))

    def decode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decodeFrac(x)[0]

    def encrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encryptFrac(np.array([x], dtype=np.float64))

    def decrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decrypt(x)[0]

    def generate_keys(self):
        self.he.keyGen()
        self.he.rotateKeyGen()

    def generate_relin_keys(self):
        self.he.relinKeyGen()

    def get_public_key(self):
        self.he.save_public_key("../key_storage/pub.key")
        with open("../key_storage/pub.key", 'rb') as f:
            return f.read()
    def get_secret_key(self):
        self.he.save_secret_key("../key_storage/secret.key")
        with open("../key_storage/secret.key", 'rb') as f:
            return f.read()
    def get_rotate_key(self):
        self.he.save_rotate_key("../key_storage/rotate.key")
        with open("../key_storage/rotate.key", "rb") as f:
            return f.read()
    def get_relin_key(self):
        self.he.save_relin_key("../key_storage/relin.key")
        with open("../key_storage/relin.key", 'rb') as f:
            return f.read()

    def load_public_key(self, key):
        with open("../key_storage/pub.key", 'wb') as f:
            f.write(key)
        self.he.load_public_key("../key_storage/pub.key")

    def load_relin_key(self, key):
        with open("../key_storage/relin.key", 'wb') as f:
            f.write(key)
        self.he.load_relin_key("../key_storage/relin.key")

    def load_secret_key(self, key):
        with open("../key_storage/secret.key", 'wb') as f:
            f.write(key)
        self.he.load_secret_key("../key_storage/secret.key")

    def load_rotate_key(self, key):
        with open("../key_storage/rotate.key", "wb") as f:
            f.write(key)
        self.he.load_rotate_key("../key_storage/rotate.key")

    def encode_matrix(self, matrix):
        """Encode a float nD-matrix in a PyPtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded
        Returns
        -------
        matrix
            nD-np.array( dtype=PyPtxt ) with encoded values
        """

        try:
            return np.array(list(map(self.encode, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
        """Decode a PyPtxt nD-matrix in a float nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyPtxt )
            matrix to be decoded
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with float values
        """
        try:
            return np.array(list(map(self.decode, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
        """Encrypt a float nD-matrix in a PyCtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=PyCtxt ) with encrypted values
        """
        try:
            return np.array(list(map(self.encrypt, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])

    def decrypt_matrix(self, matrix):
        """Decrypt a PyCtxt nD matrix in a float nD matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyCtxt )
            matrix to be decrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with plain values
        """
        try:
            return np.array(list(map(self.decrypt, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])

    def encode_number(self, number):
        return self.encode(number)

    def power(self, number, exp):
        if isinstance(number, np.ndarray):
            raise TypeError
        if exp != 2:
            raise NotImplementedError("Only square")
        s = number * number
        self.he.relinearize(s)
        self.he.rescale_to_next(s)
        return s

    def noise_budget(self, ciphertext):
        return None


class BFVPyfhel_Fractional(HE):
    def __init__(self, m, p, sec=128, int_digits=64, frac_digits=32):
        self.he = Pyfhel()
        self.he.contextGen(p, m=m, sec=sec, fracDigits=frac_digits,
                           intDigits=int_digits)

    def encode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.encodeFrac(x)

    def decode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.decodeFrac(x)

    def encrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.ecryptFrac(x)

    def decrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return self.he.deecryptFrac(x)

    def generate_keys(self):
        self.he.keyGen()

    def generate_relin_keys(self, bitCount=60, size=3):
        self.he.relinKeyGen(bitCount, size)

    def get_public_key(self):
        self.he.savepublicKey(tmp_dir.name + "/pub.key")
        with open(tmp_dir.name + "/pub.key", 'rb') as f:
            return f.read()

    def get_relin_key(self):
        self.he.saverelinKey(tmp_dir.name + "/relin.key")
        with open(tmp_dir.name + "/relin.key", 'rb') as f:
            return f.read()

    def load_public_key(self, key):
        with open(tmp_dir.name + "/pub.key", 'wb') as f:
            f.write(key)
        self.he.restorepublicKey(tmp_dir.name + "/pub.key")

    def load_relin_key(self, key):
        with open(tmp_dir.name + "/relin.key", 'wb') as f:
            f.write(key)
        self.he.restorerelinKey(tmp_dir.name + "/relin.key")

    def encode_matrix(self, matrix):
        """Encode a float nD-matrix in a PyPtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded
        Returns
        -------
        matrix
            nD-np.array( dtype=PyPtxt ) with encoded values
        """

        try:
            return np.array(list(map(self.he.encodeFrac, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
        """Decode a PyPtxt nD-matrix in a float nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyPtxt )
            matrix to be decoded
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with float values
        """
        try:
            return np.array(list(map(self.he.decodeFrac, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
        """Encrypt a float nD-matrix in a PyCtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=PyCtxt ) with encrypted values
        """
        try:
            return np.array(list(map(self.he.encryptFrac, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])

    def decrypt_matrix(self, matrix):
        """Decrypt a PyCtxt nD matrix in a float nD matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyCtxt )
            matrix to be decrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with plain values
        """
        try:
            return np.array(list(map(self.he.decryptFrac, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])

    def encode_number(self, number):
        return self.he.encode(number)

    def power(self, number, exp):
        return self.he.power(number, exp)

    def noise_budget(self, ciphertext):
        try:
            return self.he.noiseLevel(ciphertext)
        except SystemError:
            return "Can't get NB without secret key."


class CKKSTenSEAL(HE):
    def __init__(self, m=2**14, scale=2**30, qi=[60, 30, 30, 30, 60]):
        self.he = ts.context(ts.SCHEME_TYPE.CKKS, m, coeff_mod_bit_sizes=qi)
        self.he.global_scale = scale

    def encode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x

    def decode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x

    def encrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return ts.ckks_vector(self.he, [x])

    def decrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x.decrypt()[0]

    def generate_keys(self):
        self.he.generate_galois_keys()

    def get_public_key(self):
        pass

    def get_relin_key(self):
        pass

    def load_public_key(self, key):
        pass

    def load_relin_key(self, key):
        pass

    def encode_matrix(self, matrix):
        """Encode a float nD-matrix in a PyPtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded
        Returns
        -------
        matrix
            nD-np.array( dtype=PyPtxt ) with encoded values
        """

        try:
            return np.array(list(map(self.encode, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
        """Decode a PyPtxt nD-matrix in a float nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyPtxt )
            matrix to be decoded
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with float values
        """
        try:
            return np.array(list(map(self.decode, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
        """Encrypt a float nD-matrix in a PyCtxt nD-matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=PyCtxt ) with encrypted values
        """
        try:
            return np.array(list(map(self.encrypt, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])

    def decrypt_matrix(self, matrix):
        """Decrypt a PyCtxt nD matrix in a float nD matrix.
        Parameters
        ----------
        matrix : nD-np.array( dtype=PyCtxt )
            matrix to be decrypted
        Returns
        -------
        matrix
            nD-np.array( dtype=float ) with plain values
        """
        try:
            return np.array(list(map(self.decrypt, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])

    def encode_number(self, number):
        return self.encode(number)

    def power(self, number, exp):
        return number * number

    def noise_budget(self, ciphertext):
        return None


class MockHE(HE):
    """
    For testing purposes.
    """
    def __init__(self):
        pass

    def encode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x

    def decode(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x

    def encrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x

    def decrypt(self, x):
        if isinstance(x, np.ndarray):
            raise TypeError
        return x

    def generate_keys(self):
        pass

    def get_public_key(self):
        pass

    def get_relin_key(self):
        pass

    def load_public_key(self, key):
        pass

    def load_relin_key(self, key):
        pass

    def encode_matrix(self, matrix):
        try:
            return np.array(list(map(self.encode, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
        try:
            return np.array(list(map(self.decode, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
        try:
            return np.array(list(map(self.encrypt, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])

    def decrypt_matrix(self, matrix):
        try:
            return np.array(list(map(self.decrypt, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])

    def power(self, number, exp):
        return number ** exp

    def noise_budget(self, ciphertext):
        return None