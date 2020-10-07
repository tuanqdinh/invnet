# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/samples/ ./results/

rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/2010-invnet/results/samples ./results/ --exclude=checkpoints
