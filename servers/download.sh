rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/checkpoints/fnet-pix2pix-mnist-7-mixhidden-noact-4 ./results/checkpoints/

# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/2010-invnet/results/checkpoints ./results/

# rsync -av -e "ssh -i ~/Documents/Projects/_reports/klein.pem" ubuntu@3.138.236.125:~/Projects/invnet/results/samples results/  --exclude=*.npy --exclude=models
