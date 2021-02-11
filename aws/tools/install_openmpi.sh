# configure, download, and install OpenMPI

KEY_PEM_NAME=klein.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
do
  # genearte key
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'conda install -y -c conda-forge openmpi' &
  echo "Update ${i}"
 done

