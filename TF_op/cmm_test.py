""" create tf op with
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared classic_mat_mul.cc -o classic_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -undefined dynamic_lookup
"""

import tensorflow as tf
import unittest

#tf.compat.v1.disable_eager_execution()

classic_mm_module = tf.load_op_library('./classic_mat_mul.so')
#classic_mm_module = tf.load_op_library('./fast_mat_mul.so')

tf.random.set_seed(1)

a = tf.Variable(tf.random.uniform(shape=(4, 4)))
b = tf.Variable(tf.random.uniform(shape=(4, 4)))

op = classic_mm_module.ClassicMatMul(a_matrix=a, b_matrix=b)

regular = tf.matmul(a, b)

print('\n\n\nregular: ', regular)
print('\n\n\nop: ', op)
print('\n\n\nop-regular ', op-regular)



