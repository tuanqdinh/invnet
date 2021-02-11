KEY_PEM_NAME=klein.pem

cd ~/.ssh
cp ~/aws/tools/config ./
cp ~/aws/klein.pem ./

eval `ssh-agent -s`
ssh-add ${KEY_PEM_NAME}

ssh-keygen -t rsa -b 4096 -C "xxxxxx10@gmail.com" -N "" -f id_rsa