###================= setup ===================

nactors=10
resume=-1

### Train
python ./train_inv_net.py --config config/cifar10.json --resume $resume --lr 0.002 --epochs 200 --batch_size 128 --log_steps 100 --mixup_hidden

# parameters: lr, lamb

# > ../results/logs/inet-"$1".txt &

## Evaluate iResNet
resume=221
# python ./train_inv_net.py --config_file config/"$1".json --nactors $nactors --resume $resume --batch_size 512 --sample_fused

## Evaluate fusion net
fname="unet-skip-elu"
resume_g=221

# python ./train_inv_net.py --config_file config/"$1".json --nactors $nactors --resume $resume --fname $fname --resume_g $resume_g --batch_size 128 --eval_fusion
