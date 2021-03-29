for sq in 6 # 12 1 
do

for name in classic dgemm bini322 bini232 bini223 schonhage333 smirnov224 smirnov242 smirnov422 smirnov225 smirnov252 smirnov522 smirnov272 smirnov227 smirnov722 smirnov323 smirnov332 smirnov233 smirnov333 smirnov343 smirnov433 smirnov334 smirnov442 smirnov424 smirnov244 smirnov444 smirnov552 smirnov525 smirnov255 smirnov555
do

python shape_time_test.py --tf ${sq} --omp ${sq} --mm ${name} --num 1

done

done