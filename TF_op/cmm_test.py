# create tf op with
# g++ -std=c++11 -shared classic_mat_mul.cc -o classic_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -undefined dynamic_lookup

import tensorflow as tf
classic_mm_module = tf.load_op_library('./classic_mat_mul.so')

a = tf.random.uniform(shape=(2, 2))
b = tf.random.uniform(shape=(2, 2))

op = classic_mm_module.ClassicMatMul(a_matrix=a, b_matrix=b)

regular = tf.matmul(a, b)

print('\n\n\nregular: ', regular)
print('\n\n\nop: ', op)
print('\n\n\nregular-op: ', regular-op)