import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')

print(zero_out_module.zero_out([[1, 2], [3, 4]]))

"""
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)
"""
