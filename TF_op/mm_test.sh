for sq in 1 12
do

for name in dgemm bini322 schonhage333 smirnov224 smirnov225 smirnov272 smirnov323 smirnov333 smirnov334 smirnov442 smirnov444 smirnov552 smirnov555
do

python matmul_time_test.py --tf ${sq} --omp ${sq} --mm ${name}

done

done