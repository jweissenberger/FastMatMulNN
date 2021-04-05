#!/usr/bin/python

import tensorflow as tf
import time
import argparse
import numpy as np
#from openmpext import controlOMP


parser = argparse.ArgumentParser()
#parser.add_argument("--tf", type=int)
#parser.add_argument("--omp", type=int)
#parser.add_argument("--mkl", type=int)
parser.add_argument("--mm", type=str)
parser.add_argument("--num", type=int) # the number of the benchmark

args = parser.parse_args()
#ompNum = args.ompNumThread

# to change TensorFlow's threads at runtime
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(1)
'''
#print(controlOMP(args.omp))
# to change MKL's threads at runtime
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

mkl_set_num_threads(args.mkl)
print( mkl_get_max_threads() )
'''
algo_name = args.mm
epsilon_ = 1e-3
step_ = 1
fast_mm_module = tf.load_op_library('obj/%s_mat_mul.so'%algo_name)

epsilon_values = {
    'bini322': -11, 'bini232': -11, 'bini223': -11,
    'schonhage333': -5,
    'smirnov224': -7, 'smirnov242': -7, 'smirnov422': -7,
    'smirnov225': -5, 'smirnov252': -5, 'smirnov522': -5,
    'smirnov323': -5, 'smirnov332': -5, 'smirnov233': -5,
    'smirnov334': -5, 'smirnov343': -5, 'smirnov433': -5,
    'smirnov442': -5, 'smirnov424': -5, 'smirnov244': -5,
    'smirnov444': -5,
    'smirnov552': -5, 'smirnov525': -5, 'smirnov255': -5,
    'smirnov555': -5,
    'smirnov272': -3, 'smirnov227': -3, 'smirnov722': -3,
    'smirnov333': -3,
    'dgemm': -3 # placeholder
}

epsilon_ = epsilon_values.get(algo_name)+2




diff = 0
#dims = [4096] #[512, 1024, 2048, 4096, 8192]
err = np.zeros((5, 8))
k=0
for dim in range(512, 4096+512, 512):
    a = tf.Variable(tf.random.uniform(shape=(25600, 4096)), dtype=tf.float32)
    b = tf.Variable(tf.random.uniform(shape=(4096, dim)), dtype=tf.float32)
    regular = tf.matmul(a, b)
    loops = 1
    
    for j in range(5):
        diff = 0
        ep = 2**(epsilon_-j)
        for i in range(loops):

            op = fast_mm_module.FastMatMul(a_matrix=a, b_matrix=b, epsilon=ep, steps=step_, numthreads=12)
            diff += tf.norm(op - regular)/ tf.norm(regular)
            
        err[j][k] = diff/loops
        #print(f'Epsilon: 2^{epsilon_-j}')
    #print(f'Average relative error: {diff/loops}')
    print(f'Matrix size: 25600X4096X{dim}')
    
    k += 1
print(f'\n\nNumber of loops: {loops}')
print(f'Algorithm tested:{algo_name}')


outfile = f'accuracyTest/shape_test/{args.num}/{algo_name}_accuracy.dat'
with open(outfile,"w") as header:
    for j in range(8):
        for i in range(5):
            header.write(f'{str("{:.2e}".format(err[i][j])).replace("-0", "-")}' + '\t')
        header.write('\n')

'''
for 1 step of schonhage
epsilon, error
1e-1,       0.003515852615237236
1e-2,       0.0011192269157618284
1e-3,       0.1134304627776146
1e-5,       1107.096435546875
0.00390625, 0.007263661827892065
'''
