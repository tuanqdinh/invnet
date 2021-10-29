* Training
# 1. Train invertible network

python ./main.py --config_file configs/mnist.json

# 2. Generate fusion dataset 

python ./main.py --config_file configs/mnist.json -fusion --nactors 2 -sample_fusion 

# 3. Train fusion network

python ./main.py --config_file configs/mnist.json -fusion --nactors 2 

# 4. Evaluate the fusion network

python ./main.py --config_file configs/mnist.json -fusion --nactors 2 -test_fusion