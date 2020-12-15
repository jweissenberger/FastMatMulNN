
for size in 512 1024 2048 4096 8192
do
for name in bini322 dgemm classic schonhage333 smirnov224 smirnov225 smirnov272 smirnov323 smirnov333 smirnov334 smirnov442 smirnov444 smirnov552 smirnov555
do

python tensorboard_test.py --layers 4 --nodes ${size} --epochs 5 --bs ${size} --mm ${name} --logdir all_threads_${name}${size}

done
done
