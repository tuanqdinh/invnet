###================= setup ===================

nactors=4
resume=100

### Train
# python ./train_inv_net.py --config config/cifar10.json --resume $resume --lr 0.001 --epochs 100 --batch_size 128 --log_steps 100 --mixup_hidden

# parameters: lr, lamb

# > ../results/logs/inet-"$1".txt &

## Evaluate iResNet
# python ./train_inv_net.py --config_file config/mnist.json --nactors $nactors --resume $resume --batch_size 512 --lr 0.05 --mixup_hidden --epochs 50 --sample_fused #--eval_inv


python ./adv_robustness.py --config_file config/cifar10.json --nactors $nactors --batch_size 128 --sample_fused


## Evaluate fusion net
fname="pix2pix"
resume_g=200

# python ./train_inv_net.py --config_file config/mnist.json --nactors $nactors --resume $resume --fname $fname --resume_g $resume_g --batch_size 128 --eval_fusion --flag_test --mixup_hidden --noActnorm

