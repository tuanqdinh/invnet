# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' ./src tuandinh@128.104.158.78:~/Documents/Projects/2010-invnet/

# rsync -av -e "ssh -i ~/Documents/Projects/_reports/klein.pem" ./src ubuntu@3.138.236.125:~/Projects/invnet/
# —progress -e

scp -r 'ssh -A dthai@swarm2.css.umass.edu ssh' ./codednet/ps_pytorch  dthai@gypsum.cs.umass.edu:/mnt/nfs/scratch1/dthai/Projects/codednet/
