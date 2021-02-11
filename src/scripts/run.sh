

nactors=$1

fname="pix2pix"
iname="mnist-7-mixhidden-act"
dataset="mnist"
init=2
resume=100 #151

# python ./train_inv_net.py --config_file config/mnist.json --nactors $nactors --resume $resume --batch_size 512 --mixup_hidden --sample_fused

resume_g=0
# python ./train_fusion_gan_net.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --init_ds $init --epochs 50 --lr 0.002 --log_steps 200

resume_g=50
python ./test_fusion_gan_net.py --fname $fname --iname $iname --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds $init --mixup_hidden --flag_test
