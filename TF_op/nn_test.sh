outpath=./output_all_threads_fastmm_gradient_regular_bini_dgemm
mkdir $outpath


for name in regular bini322 dgemm
do
for layer in 4 16 32 64 128
do
for node in 256 512 1024 2048 4096 8192 16384
do
for bs in 64 128 256 512 1024 2048 4096 8192 16384
do

python custom_op_nn.py -u --layers ${layer} --nodes ${node} --epochs 5 --bs ${bs} --mm ${name} \
                       > ${outpath}/layer_${layer}_nodes_${node}_bs_${bs}_mm_${name}.log

done
done
done
done