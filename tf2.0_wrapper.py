# Tensorflow 2.0
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Linear(Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        # ***************************************
        # this is the line we could switch with our new matmul
        return tf.matmul(inputs, self.w) + self.b

class ComputeSum(Layer):
    """Returns the sum of the inputs."""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                 trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

class Dropout(Layer):

    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    @tf.function
    def call(self, inputs, training=None):
        # Note that the tf.function decorator enables use
        # to use imperative control flow like this `if`,
        # while defining a static graph!
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

if __name__ == '__main__':
    print("Need TF2.0")
