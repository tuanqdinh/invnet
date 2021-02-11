#!/bin/bash
#=================== setup ===================
fname="pix2pix"
# iname="mnist-7-mixhidden-noact"
iname="cifar10-robustness"


# inet-cifar10-9-vanilla-noact
###================= mnist ===================
# dataset="mnist"
# init=2
# nactors=10
# resume=100 #151
# resume_g=31

####================= Fashion ================
# dataset="fashion"
# init=2
# nactors=4
# resume=201 #121
# resume_g=0

####================= Cifar10 ================
dataset="cifar10"
init=1
nactors=4
resume_g=0

####================= Train scripts ===========
# sleep 30m ; echo "5 minutes complete"

#default norm: instance
# python ./train_fnet_unet.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --init_ds $init --epochs 200 --lr 0.005
python ./train_fusion_gan_net.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --init_ds $init --epochs 100 --lr 0.002 --log_steps 200


# python ./train_distill_net.py --fname distill-bn --iname $iname --dataset $dataset --nactors 4 --resume $resume --resume_g $resume_g --init_ds $init --epochs 200 --lr 0.005 --nonlin elu --noActnorm --nBlocks 11 11 11

####================= Test scripts ===========
# python ./test_fnet_unet.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds $init --flag_test --nonlin elu --nBlocks 9 9 9

# python ./test_fusion_gan_net.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds $init --mixup_hidden --noActnorm --flag_test #--nBlocks 9 9 9

####================= Test scripts ===========
# python ./curve_fnet_unet.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume $resume --init_ds $init --batch_size 512 --niter 30