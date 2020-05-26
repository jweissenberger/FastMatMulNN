#!/usr/bin/python

import tensorflow as tf
import time

# to change MKL's threads at runtime
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

fast_mm_module = tf.load_op_library('./schonhage_mat_mul.so')

custom_time = 0
regular_time = 0

print( mkl_get_max_threads() )
mkl_set_num_threads(1)
print( mkl_get_max_threads() )

diff = 0

dim = 2000
loops = 20
for i in range(loops):

    a = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)
    b = tf.Variable(tf.random.uniform(shape=(dim, dim)), dtype=tf.float32)

    t1 = time.time()
    op = fast_mm_module.FastMatMul(a_matrix=b, b_matrix=a, epsilon=1e-5, steps=1)
    t2 = time.time()

    t3 = time.time()
    regular = tf.matmul(a, b)
    t4 = time.time()

    custom_time += t2-t1
    regular_time += t4-t3

    diff += tf.norm(op - regular)/ tf.norm(regular)

avg_custom = custom_time/loops
avg_reg = regular_time/loops
print(f'\n\nNumber of loops:{loops}')
print(f'Average custom time: {avg_custom}')
print(f'Average regular time: {avg_reg}')
print(f'Times faster: {avg_reg/avg_custom}')
print(f'Average relative error: {diff/loops}')

'''
for 1 step of schonhage
epsilon, error
1e-1,       0.003515852615237236
1e-2,       0.0011192269157618284
0.00390625, 0.007263661827892065
'''
