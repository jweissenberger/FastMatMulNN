outpath=./output_first
mkdir $outpath


for name in regular bini_mat_mul
do
for layer in 4 16 #32 64 128
do
for node in 128 256 #512 1024 2048 4096 8192
do
for bs in 64 128 #256 512 1024 #2048 4096 8192 16384
do

python custom_op_nn.py --layers ${layer} --nodes ${node} --epochs 5 --bs ${bs} --mm ${name} \
                       > ${outpath}/layer_${layer}_nodes_${node}_bs_${bs}_mm_${name}.log

done
done
done
done