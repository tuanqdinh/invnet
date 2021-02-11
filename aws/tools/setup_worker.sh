KEY_PEM_NAME=klein.pem
export WORKERS_COUNT=`wc -l < hosts`

i=$1 #from 2
cd ~/.ssh
scp -i ${KEY_PEM_NAME} -r ~/aws deeplearning-worker${i}:~/
ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'cd ~/aws; bash tools/setup_key_worker.sh; cd ~/aws; bash tools/install_mpi.sh; ' 
echo "Done worker ${i}"