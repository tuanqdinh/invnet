export WORKERS_COUNT=`wc -l < hosts`

for i in $(seq 2 $WORKERS_COUNT);
do
  # setup from ps
  scp ~/aws/src/*.py deeplearning-worker${i}:~/aws/src/
  # scp ~/aws/src/codednet/invnet/* deeplearning-worker${i}:~/aws/src/codednet/invnet/
  echo "Done ${i}"
doneh 