# Results_Verification_PPML
A secure DNN inference service with dynamic verification

First, we should install some packages as follows:

1. pip install torch, torchvision
2. pip install ecdsa==0.19.0
3. git clone --recursion https://github.com/ibarrond/Pyfhel.git
    <br>
    cd Pyfhel
    <br>
    vim pyproject.toml-------modify SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT='OFF'
    <br>
    pip install .
    <br>
    cd Pyfhel/Pyfhel----------delete __init__.py 
