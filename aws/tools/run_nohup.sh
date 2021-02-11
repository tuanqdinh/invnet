

for i in $(seq 4 12)
do
    nohup bash tools/setup_worker.sh $i > log_"$i".txt &
done