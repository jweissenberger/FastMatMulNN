#!/usr/bin/python

import tensorflow as tf
import time

# to change MKL's threads at runtime
#import ctypes
#mkl_rt = ctypes.CDLL('libmkl_rt.so')
#mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
#mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
#print( mkl_get_max_threads() )
#mkl_set_num_threads(1)

algo_name = 'bini322'
epsilon_ = 1e-3
step_ = 1
fast_mm_module = tf.load_op_library('obj/%s_mat_mul.so'%algo_name)

custom_time = 0
regular_time = 0


# to change TensorFlow's threads at runtime
#tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.config.threading.set_inter_op_parallelism_threads(1)

diff = 0
dims = [512, 1024, 2048, 4096, 8192]
for dim in dims:
    loops = 3
    for i in range(loops):

        a = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)
        b = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)

        t1 = time.time()
        op = fast_mm_module.FastMatMul(a_matrix=a, b_matrix=b, epsilon=epsilon_, steps=step_)
        t2 = time.time()

        t3 = time.time()
        regular = tf.matmul(a, b)
        t4 = time.time()

        custom_time += t2-t1
        regular_time += t4-t3

        diff += tf.norm(op - regular)/ tf.norm(regular)

    avg_custom = custom_time/loops
    avg_reg = regular_time/loops
    print(f'\n\nNumber of loops: {loops}')
    print(f'Matrix size: {dim}X{dim}')
    print(f'Epsilon: {epsilon_}')
    print(f'Steps: {step_}')
    print(f'Algorithm tested:{algo_name}')
    print(f'Average custom time: {avg_custom}')
    print(f'Average regular time: {avg_reg}')
    print(f'Times faster: {avg_reg/avg_custom}')
    print(f'Average relative error: {diff/loops}')

'''
for 1 step of schonhage
epsilon, error
1e-1,       0.003515852615237236
1e-2,       0.0011192269157618284
1e-3,       0.1134304627776146
1e-5,       1107.096435546875
0.00390625, 0.007263661827892065
'''
