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

dim = 500
loops = 2
for i in range(loops):

    a = tf.Variable(tf.random.uniform(shape=(dim, dim)))
    b = tf.Variable(tf.random.uniform(shape=(dim, dim)))

    t1 = time.time()
    op = fast_mm_module.FastMatMul(a_matrix=b, b_matrix=a, epsilon=1e-1, steps=2)
    t2 = time.time()

    t3 = time.time()
    regular = tf.matmul(a, b)
    t4 = time.time()

    custom_time += t2-t1
    regular_time += t4-t3
    print(op, "\n\n\n")
    print(regular, "\n\n\n")

    print(op-regular)

    diff += tf.norm(op - regular)/ tf.norm(regular)

print(f'\n\nNumber of loops:{loops}')
print(f'Average custom time: {custom_time/loops}')
print(f'Average regular time: {regular_time/loops}')
print(f'Average relative error: {diff/loops}')
