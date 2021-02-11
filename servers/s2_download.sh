rsync -av -e 'ssh -A dthai@swarm2.cs.umass.edu ssh' dthai@gypsum.cs.umass.edu:/mnt/nfs/scratch1/dthai/Projects/codednet/parm/train/save ./parm/train/ --exclude=.pth

# rsync -av -e 'ssh -A dthai@swarm2.cs.umass.edu ssh' dthai@gypsum.cs.umass.edu:/mnt/nfs/scratch1/dthai/Projects/codednet/results/samples ./results/
