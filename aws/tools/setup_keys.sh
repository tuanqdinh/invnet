KEY_PEM_NAME=klein.pem
export WORKERS_COUNT=`wc -l < hosts`

cd ~/.ssh
eval `ssh-agent -s`
ssh-add ${KEY_PEM_NAME}

# connection 2 ways
for i in $(seq 4 12);
do
  # setup from ps
  scp -i ${KEY_PEM_NAME} id_rsa.pub deeplearning-worker${i}:~/.ssh/id_ps.pub
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'cd ~/.ssh; cat id_ps.pub >> authorized_keys;' 
  scp -i ${KEY_PEM_NAME} deeplearning-worker${i}:~/.ssh/id_rsa.pub id_client.pub 
  cat id_client.pub >> authorized_keys
  echo "Done writing public ${i}"
done