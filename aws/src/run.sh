cd ..
sh src/update_src.sh
cd src

/home/ubuntu/anaconda3/envs/pytorch_p36/bin/mpirun -mca btl_tcp_if_include 172.31.0.0/20 -n $1 --hostfile ../hosts_address --use-hwthread-cpus \
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python main.py --epochs 1000 --enable-gpu True

# -mca btl_tcp_if_include 172.31.0.0/20
#/home/ubuntu/anaconda3/envs/pytorch_p36/bin/