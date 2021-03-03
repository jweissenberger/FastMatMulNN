
for name in bini322 dgemm schonhage333 smirnov224 smirnov225 smirnov272 smirnov323 smirnov333 smirnov334 smirnov442 smirnov444 smirnov552 smirnov555
do


python accuracy_test.py --layers 4 --nodes 300 --epochs 50 --bs 300 --mm ${name} --num_threads 12

done
