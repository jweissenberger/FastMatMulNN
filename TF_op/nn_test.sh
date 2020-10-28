outpath=./output_one_thread_fastmm_gradient_regular_bini_dgemm
mkdir $outpath


for size in 512 1024 4096 8192
do
for name in regular bini322 dgemm
do

python custom_op_nn.py --layers 4 --nodes ${size} --epochs 5 --bs ${size} --mm ${name} \
                       > ${outpath}/layer_${layer}_nodes_${node}_bs_${bs}_mm_${name}.log

done
done
done
done
