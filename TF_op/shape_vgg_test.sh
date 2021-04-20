for sq in 6 # 12 1
do

for shape in bs,25,4 bs,4,25 25,bs,4 bs,4,4 4,bs,4 bs,4,1 bs,1,4 4,bs,1
do

mkdir matmalTest/par_${sq}/${shape}/2

for name in classic dgemm bini322 bini232 bini223 schonhage333 smirnov224 smirnov242 smirnov422 smirnov225 smirnov252 smirnov522 smirnov272 smirnov227 smirnov722 smirnov323 smirnov332 smirnov233 smirnov333 smirnov343 smirnov433 smirnov334 smirnov442 smirnov424 smirnov244 smirnov444 smirnov552 smirnov525 smirnov255 smirnov555
do

python shape_vgg_time_test.py --tf ${sq} --omp ${sq} --mm ${name} --shape ${shape} --num 2

done

cd matmalTest
python convert.py --shape ${shape}
cd ../

done

done

