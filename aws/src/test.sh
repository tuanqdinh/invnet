cd ..
sh src/update_src.sh
cd src

/home/ubuntu/anaconda3/envs/pytorch_p36/bin/mpirun -mca btl_tcp_if_include 172.31.0.0/20 -n $1 --hostfile host_test \
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python test_mpi.py

# -mca btl_tcp_if_include 172.31.0.0/20 
# 