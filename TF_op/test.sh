outpath=./output_first
mkdir $outpath


for name in regular bini_mat_mul
do

python custom_op_nn.py -u --layers 2 --nodes 30 --epochs 5 --bs 64 --mm ${name} \
                       > ${outpath}/test.log

done