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
parser.add_argument("--shape", type=str)
parser.add_argument("--num", type=int) # the number of the benchmark

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

shapes = args.shape.split(',')
print(shapes)
assert len(shapes) == 3
reg_dim = []
for d in shapes:
    if d == 'bs':
        bs_dim = shapes.index(d)
        reg_dim.append(0)
    else:
        reg_dim.append(int(d))
assert reg_dim[bs_dim] == 0

epsilon_values = {
    'bini322': -11, 'bini232': -11, 'bini223': -11,
    'schonhage333': -6,
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

epsilon_ = epsilon_values.get(algo_name)


loops = 3
times = np.zeros((loops, 4))
k=0

for dim in range(1024, 4096+1024, 1024):
    
    if bs_dim == 0:
        a = tf.Variable(tf.random.uniform(shape=(dim, reg_dim[1] * 1024)), dtype=tf.float32)
        b = tf.Variable(tf.random.uniform(shape=(reg_dim[1] * 1024, reg_dim[2] * 1024)), dtype=tf.float32)
    elif bs_dim == 1:
        a = tf.Variable(tf.random.uniform(shape=(reg_dim[0] * 1024, dim)), dtype=tf.float32)
        b = tf.Variable(tf.random.uniform(shape=(dim, reg_dim[2] * 1024)), dtype=tf.float32)
    elif bs_dim == 2:
        a = tf.Variable(tf.random.uniform(shape=(reg_dim[0] * 1024, reg_dim[1] * 1024)), dtype=tf.float32)
        b = tf.Variable(tf.random.uniform(shape=(reg_dim[1] * 1024, dim)), dtype=tf.float32)

    if algo_name == 'classic':
        t1 = time.time()
        regular = tf.matmul(a, b)
        t2 = time.time()
        times[0][k] = 1e-9 * 2 * (dim**3) / (t2-t1)
    
    else:
        for i in range(loops):

            t1 = time.time()
            op = fast_mm_module.FastMatMul(a_matrix=a, b_matrix=b, epsilon=2**epsilon_, steps=step_, numthreads=args.omp)
            t2 = time.time()

            times[i][k]=  1e-9 * 2 * (dim**3) / (t2-t1)
        
    #print(f'Average relative error: {diff/loops}')
    if bs_dim == 0:
        print(f'Matrix size: {dim}X{reg_dim[1]}kX{reg_dim[2]}k')
    elif bs_dim == 1:
        print(f'Matrix size: {reg_dim[0]}kX{dim}X{reg_dim[2]}k')
    elif bs_dim == 2:
        print(f'Matrix size: {reg_dim[0]}kX{reg_dim[1]}kX{dim}')
    k += 1

print(f'\n\nNumber of loops: {loops}')
print(f'Algorithm tested:{algo_name}')
print(f'\n\n')

computeMethod = 'seq'
if args.omp == 1:
    computeMethod = 'seq'
if args.omp == 12:
    computeMethod = 'par_12'
if args.omp == 6:
    computeMethod = 'par_6'

if algo_name != 'classic':
    medianTimes = np.median(times, axis=0)
outfile = f'matmalTest/{computeMethod}/{args.shape}/{args.num}/{algo_name}_shape_vgg_test.dat'
with open(outfile,"w") as header:
    
    if algo_name == 'classic':
        for j in range(4):
            if times[0][j] > 1:
                header.write(f'{str("{:.3e}".format(times[0][j])).replace("+0","")}' + '\t')
            else:
                header.write(f'{str("{:.2e}".format(times[0][j])).replace("-0","-")}' + '\t')
    
    else:
        for i in range(loops):
            for j in range(4):
                if times[i][j] > 1:
                    header.write(f'{str("{:.3e}".format(times[i][j])).replace("+0", "")}' + '\t')
                else:
                    header.write(f'{str("{:.2e}".format(times[i][j])).replace("-0", "-")}' + '\t')
            header.write('\n')

        for j in range(4):
            if medianTimes[j] > 1:
                header.write(f'{str("{:.3e}".format(medianTimes[j])).replace("+0","")}' + '\t')
            else:
                header.write(f'{str("{:.2e}".format(medianTimes[j])).replace("-0","-")}' + '\t')
