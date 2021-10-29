# Coded-InvNet for Resilient Prediction Serving Systems

In this repository, we provide the source code for the Coded-InvNet method. 

Here is an example of the script on MNIST, using i-ResNet-64 for the invertible network and PixPix for the fusion network, with k=2.

## 1. Train invertible network

python ./main.py --config_file configs/mnist.json

## 2. Generate fusion dataset 

python ./main.py --config_file configs/mnist.json -fusion --nactors 2 -sample_fusion 

## 3. Train fusion network

python ./main.py --config_file configs/mnist.json -fusion --nactors 2 

## 4. Evaluate the fusion network

python ./main.py --config_file configs/mnist.json -fusion --nactors 2 -test_fusion

