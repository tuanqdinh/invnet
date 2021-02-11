cd ..
sh src/update_src.sh
cd src

/home/ubuntu/anaconda3/envs/pytorch_p36/bin/mpirun -mca btl_tcp_if_include 172.31.0.0/20 -n $1 --hostfile ../hosts_address --use-hwthread-cpus \
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python main_pm.py --epochs 1000 --enable-gpu True



# mpirun -n $1 --hostfile host_local \
# python main_pm.py --epochs 1

# -mca btl_tcp_if_include 172.31.0.0/20
#/home/ubuntu/anaconda3/envs/pytorch_p36/bin/