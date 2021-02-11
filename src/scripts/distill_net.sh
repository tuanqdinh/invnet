iname="elu-mixup"
fname="distill-bn"

# if ; then
# elif ; then
# else
# fi
# tensorboard --logdir=runs
###================= mnist =================

#### MNIST ###
dataset="mnist"
init=2
nactors=10
resume=151
resume_g=0
lr=0.002
epochs=300
lambda_L1=1
lambda_distill=50
#36690
####================= Fashion ###=================
# dataset="fashion"
# init=2
# nactors=2
# resume=201
# resume_g=0
# epochs=150
# lr=0.0002
# lambda_L1=1
# lambda_distill=100

# nohup sh scripts/distill_net.sh  > ../results/logs/fnet_distill_fashion_mixup_4.txt &
# alias tf='tail -n 100 ../results/logs/fnet_distill_fashion_mixup_4.txt'
####================= Cifar10 ###=================
# dataset="cifar10"
# init=1
# nactors=4
# resume=200
# resume_g=140
# lr=0.0001
# epochs=200
# lambda_L1=1
# lambda_distill=100
#11098
# nohup sh scripts/distill_net.sh  > ../results/logs/fnet_distill_cifar10_mixup_4.txt &
# alias tf='tail -n 100 ../results/logs/fnet_distill_cifar10_mixup_4.txt'

# rm -r ../results/runs
python ./train_integrated_net.py --dataset $dataset --fname $fname --iname $iname --nactors $nactors --resume $resume --nonlin elu --noActnorm --nBlocks 7 7 7 --resume_g $resume_g --init_ds $init --epochs $epochs --lr $lr --lambda_L1 $lambda_L1 --lambda_distill $lambda_distill
