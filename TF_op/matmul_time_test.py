#!/usr/bin/python

import tensorflow as tf
import time
import argparse
import numpy as np
#from openmpext import controlOMP


parser = argparse.ArgumentParser()
parser.add_argument("--tf", type=int)
parser.add_argument("--omp", type=int)
#parser.add_argument("--mkl", type=int)
parser.add_argument("--mm", type=str)

args = parser.parse_args()
#ompNum = args.ompNumThread

# to change TensorFlow's threads at runtime
tf.config.threading.set_intra_op_parallelism_threads(args.tf)
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
step_ = 1
fast_mm_module = tf.load_op_library('obj/%s_mat_mul.so'%algo_name)


epsilon_values = {
    'bini322': -11,
    'schonhage333': -5,
    'smirnov224': -7,
    'smirnov225': -5,
    'smirnov323': -5,
    'smirnov334': -5,
    'smirnov442': -5,
    'smirnov444': -5,
    'smirnov552': -5,
    'smirnov555': -5,
    'smirnov272': -3,
    'smirnov333': -3,
    'dgemm': -3 # placeholder
}

epsilon_ = epsilon_values.get(algo_name)


loops = 5
times = np.zeros((loops, 16))
k=0

for dim in range(512, 8192+512, 512):
    
    '''
    a = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)
    b = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)
    t3 = time.time()
    regular = tf.matmul(a, b)
    t4 = time.time()
    regular_time = t4-t3
    '''
    
    for i in range(loops):

        a = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)
        b = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)

        t1 = time.time()
        op = fast_mm_module.FastMatMul(a_matrix=a, b_matrix=b, epsilon=2**epsilon_, steps=step_, numthreads=args.omp)
        t2 = time.time()

        times[i][k]=  1e-9 * 2 * (dim**3) / (t2-t1)
        
    #print(f'Average relative error: {diff/loops}')
    print(f'Matrix size: {dim}X{dim}')
    k += 1

print(f'\n\nNumber of loops: {loops}')
print(f'Algorithm tested:{algo_name}')
print(f'\n\n')

computeMethod = 'seq'
if args.omp == 12:
    computeMethod = 'par'
medianTimes = np.median(times, axis=0)
outfile = f'matmalTest/{computeMethod}/{algo_name}_test.dat'
with open(outfile,"w") as header:
    for i in range(loops):
        for j in range(16):
            header.write(f'{str("{:.3e}".format(times[i][j])).replace("+0", "")}' + '\t')
        header.write('\n')

    for j in range(16):
        header.write(f'{str("{:.3e}".format(medianTimes[j])).replace("+0","")}' + '\t')
