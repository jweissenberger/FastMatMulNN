outpath=./output_all_threads_fastmm_gradient_regular_bini_dgemm
mkdir $outpath


for layer in 4 8 16
do
for node in 256 1024 4096 8192
do
for bs in 256 1024 4096 8192
do
for name in regular bini322 dgemm
do

python custom_op_nn.py --layers ${layer} --nodes ${node} --epochs 5 --bs ${bs} --mm ${name} \
                       > ${outpath}/layer_${layer}_nodes_${node}_bs_${bs}_mm_${name}.log

done
done
done
done