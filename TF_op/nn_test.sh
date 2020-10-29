outpath=./output_one_thread_fastmm_gradient_regular_bini_dgemm
mkdir $outpath


for size in 512 1024 2048 4096 8192
do
for name in regular bini322 dgemm
do

python custom_op_nn.py --layers 4 --nodes ${size} --epochs 5 --bs ${size} --mm ${name} \
                       > ${outpath}/layer_4_nodes_${size}_bs_${size}_mm_${name}.log

done
done
done
done
