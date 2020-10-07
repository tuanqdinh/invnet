#!/bin/bash
#=================== setup ===================
fname="pix2pix"
iname="cifar10-9-vanilla-noact"

# inet-cifar10-9-vanilla-noact
###================= mnist ===================
# dataset="mnist"
# init=2
# nactors=4
# resume=201 #151
# resume_g=200

####================= Fashion ================
# dataset="fashion"
# init=2
# nactors=4
# resume=201 #121
# resume_g=0

####================= Cifar10 ================
dataset="cifar10"
init=1
nactors=10
resume=221
resume_g=151

####================= Train scripts ===========
# sleep 30m ; echo "5 minutes complete"

#default norm: instance
# python ./train_fnet_unet.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --init_ds $init --epochs 200 --lr 0.005
# python ./train_fusion_gan_net.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --init_ds $init --epochs 200 --lr 0.0002


# python ./train_distill_net.py --fname distill-bn --iname $iname --dataset $dataset --nactors 4 --resume $resume --resume_g $resume_g --init_ds $init --epochs 200 --lr 0.005 --nonlin elu --noActnorm --nBlocks 11 11 11

####================= Test scripts ===========
# python ./test_fnet_unet.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds $init --flag_test --nonlin elu --nBlocks 9 9 9

python ./test_fusion_gan_net.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds $init --batch_size 64 --lamb 100 --noActnorm --nBlocks 9 9 9

####================= Test scripts ===========
# python ./curve_fnet_unet.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume $resume --init_ds $init --batch_size 512 --niter 30

####================= Results ================
# U-NET-100
# MNIST-4
# Train: L1 0.0848 Match: 0.9365
# Test: L1 0.1428 Match: 0.8002

# elu-bn
# test: L1 0.0432 Match: 0.9484
# val: L1 0.0393 Match: 0.9559
# train: L1: 0.0345 Match: 0.9764

# MNIST-2
# Train: L1 0.1007 Match: 0.9940
# Test: L1 0.1357 Match: 0.9784

# U-NET-L2-151
# MNIST-4
# Train: L2 0.0131 Match: 0.8871
# Test: L2 0.0441 Match: 0.7216
# MNIST-2
# Train: L2 0.0189 Match: 0.9903
# Test: L1 0.0395 Match: 0.9712

# Pix2Pix-100
# MNIST-4
# Train (): GAN: 0.2802 L1 0.0855 Match: 0.9312
# Test (2500): GAN: 0.3138 L1 0.1446 Match: 0.7899
# MNIST-2
# Train: GAN: 0.2780 L1 0.1019 Match: 0.9934
# Test: GAN: 0.2801 L1 0.1381 Match: 0.9758


# Overfit-L1
# MNIST-4
# Train: L1 0.0322 (0.0194) Match: 0.9466
# L1 0.0190 Match: 0.9457
# Test: L1 0.1991 Match: 0.5125
# MNIST-2
# Train:
# Test:

# only batch_size different, why the train losses are different? - not shuffle
# tanh loss, not original

# Overfit-L2
# MNIST-4
# Train: L2(0.0011) 0.0041 Match: 0.8958
## batch_size: 0.0011 Match: 0.9172
# Test: L2: 0.0900 Match: 0.4464
# MNIST-2
# Train:
# Test:

# NEGATIVE LR
# 2- cifar10: GAN: 0.2454 L1 10.2802 Match: 0.9795
# test: GAN: 0.4396 L1 17.4630 Match: 0.9491
# Epoch[200](570/586): D-real: 0.1756 D-fake: 0.2584 G-GAN: 0.2728 G-L1 9.7168

# Fashion-4
# Mixup: Train Set L1 0.0829 Match: 0.7900
