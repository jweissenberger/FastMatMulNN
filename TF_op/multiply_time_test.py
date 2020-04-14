#!/usr/bin/python

import tensorflow as tf
import time

# to change MKL's threads at runtime
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

fast_mm_module = tf.load_op_library('./fast_mat_mul.so')

custom_time = 0
regular_time = 0

print( mkl_get_max_threads() )
mkl_set_num_threads(1)
print( mkl_get_max_threads() )

loops = 100
for i in range(loops):

    a = tf.Variable(tf.random.uniform(shape=(1000, 1000)))
    b = tf.Variable(tf.random.uniform(shape=(1000, 1000)))

    t1 = time.time()
    op = fast_mm_module.FastMatMul(a_matrix=a, b_matrix=b)
    t2 = time.time()

    t3 = time.time()
    regular = tf.matmul(a, b)
    t4 = time.time()

    custom_time += t2-t1
    regular_time += t4-t3

print(f'\n\nNumber of loops:{loops}')
print(f'Average custom time: {custom_time/loops}')
print(f'Average regular time: {regular_time/loops}')
