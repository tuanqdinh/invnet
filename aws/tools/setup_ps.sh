KEY_PEM_NAME=klein.pem


cd ~/aws
sudo bash -c "cat hosts >> /etc/hosts"
cp /tools/config ~/.ssh/
cp klein.pem ~/.ssh/

bash ./tools/install_mpi.sh
cd ~


cd ~/.ssh
eval `ssh-agent -s`
ssh-add ${KEY_PEM_NAME}
ssh-keygen -t rsa -b 4096 -C "xxxxxx10@gmail.com" -N '' -f id_rsa
