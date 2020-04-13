#!/usr/bin/python

import tensorflow as tf
import time


fast_mm_module = tf.load_op_library('./fast_mat_mul.so')

custom_time = 0
regular_time = 0

loops = 1000
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
