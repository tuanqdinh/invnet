KEY_PEM_NAME=/mnt/nfs/scratch1/dthai/Projects/codednet/ps_pytorch/klein.pem
PUB_IP_ADDR=$1

ssh -o "StrictHostKeyChecking no" ubuntu@${PUB_IP_ADDR}
scp -i ${KEY_PEM_NAME} -r ../aws ubuntu@${PUB_IP_ADDR}:~/
scp -i ${KEY_PEM_NAME} hosts hosts_address config ubuntu@${PUB_IP_ADDR}:~/
