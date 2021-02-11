KEY_PEM_NAME=klein.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
do
  scp -i ${KEY_PEM_NAME} -r ~/src deeplearning-worker${i}:~/aws/
  echo "Done writing public key to worker: deeplearning-worker${i}"
 done