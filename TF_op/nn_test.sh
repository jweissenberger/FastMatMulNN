
for size in 512 1024 2048 4096 8192
do
for name in bini322 dgemm classic schonhage333 smirnov224 smirnov225 smirnov272 smirnov323 smirnov333 smirnov334 smirnov442 smirnov444 smirnov552 smirnov555
do
for thread in 1 12
do

python tensorboard_test.py --layers 4 --nodes ${size} --epochs 5 --bs ${size} --mm ${name} --logdir FINAL_${thread}_thread_${name}${size} --num_threads ${thread}

done
done
done


#python tensorboard_test.py --layers 4 --nodes 512 --epochs 5 --bs 512 --mm smirnov442 --logdir test