outpath=./output_12_thread_fastmm_gradient_regular_bini_dgemm
mkdir $outpath

export OMP_NUM_THREADS='12'

for size in 512 1024 2048 4096 8192
do
for name in bini322 dgemm classic schonhage333 smirnov224 smirnov225 smirnov272 smirnov323 smirnov333 smirnov334 smirnov442 smirnov444 smirnov552 smirnov555
do

python tensorboard_test.py --layers 4 --nodes ${size} --epochs 5 --bs ${size} --mm ${name} --logdir TwelveThread_${name}${size}\
                       > ${outpath}/layer_4_nodes_${size}_bs_${size}_mm_${name}.log

done
done
