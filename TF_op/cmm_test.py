import tensorflow as tf
classic_mm_module = tf.load_op_library('./classic_mat_mul.so')

print(classic_mm_module.ClassicMatMul(tf.ones(shape=(2,2)), tf.zeros(shape=(2,2))))

