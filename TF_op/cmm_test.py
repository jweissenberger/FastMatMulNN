#!/usr/bin/python

import tensorflow as tf
import unittest

#tf.compat.v1.disable_eager_execution()

#classic_mm_module = tf.load_op_library('./classic_mat_mul.so')
fast_mm_module = tf.load_op_library('./fast_mat_mul.so')

#tf.random.set_seed(1)



a = tf.Variable(tf.random.uniform(shape=(800, 800)))
b = tf.Variable(tf.random.uniform(shape=(800, 800)))

#op = classic_mm_module.ClassicMatMul(a_matrix=a, b_matrix=b)

print("looping:")
for i in range(100):
    op = fast_mm_module.FastMatMul(a_matrix=a, b_matrix=b)

# regular = tf.matmul(a, b)
# 
# print('\n\n\nregular: ', regular)
# print('\n\n\nop: ', op)
# print('\n\n\nop-regular ', op-regular)



