import time
import os

import numpy as np

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops import array_ops, gen_math_ops, math_ops

import tensorflow as tf
from tensorflow import keras

layers = VersionAwareLayers()


@tf.RegisterGradient("FastMatMul")
def _Fast_MatMul_grad(op, grad):
    bt = array_ops.transpose(op.inputs[1])
    at = array_ops.transpose(op.inputs[0])
    grad_a = fast_mm_module.FastMatMul(a_matrix=grad, b_matrix=bt, epsilon=1e-2, steps=1, numthreads=num_threads)
    grad_b = fast_mm_module.FastMatMul(a_matrix=at, b_matrix=grad, epsilon=1e-2, steps=1, numthreads=num_threads)
    # a = math_ops.conj(op.inputs[0])
    # b = math_ops.conj(op.inputs[1])
    # grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
    # grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    return grad_a, grad_b


class Fast_Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, activation='relu'):
        super(Fast_Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

        self.epsilon = epsilon_values.get(mm_algo, 1e-2)

        self.activation = activation

num_threads = 12
mm_algo = 'smirnov444'
epsilon_values = {
    'bini322': 2**-11,
    'schonhage333': 2**-5,
    'smirnov224': 2**-7,
    'smirnov225': 2**-5,
    'smirnov272': 2**-3,
    'smirnov323': 2**-5,
    'smirnov333': 2**-3,
    'smirnov334': 2**-5,
    'smirnov442': 2**-5,
    'smirnov444': 2**-5,
    'smirnov552': 2**-5,
    'smirnov555': 2**-5
}

if __name__ == '__main__':

    fast_mm_module = tf.load_op_library(f'obj/{mm_algo}_mat_mul.so')
    epochs = 2
    batch_size = 256

    y_train = tf.ones([batch_size])
    x_train = tf.ones([batch_size, 25088])

    model_input = layers.Input(shape=25088)
    # x = layers.Dense(4096, activation='relu', name='fc1')(x)
    fast_layer1 = Fast_Linear(units=4096, input_dim=25088, activation='relu')
    x = fast_layer1()

    # x = layers.Dense(4096, activation='relu', name='fc2')(x)
    fast_layer2 = Fast_Linear(units=4096, input_dim=4096, activation='relu')
    x = fast_layer2(x)

    # imagenet_utils.validate_activation('softmax', weights)

    # x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    fast_output_layer = Fast_Linear(units=1000, input_dim=4096, activation='softmax')
    x = fast_output_layer(x)

    model = training.Model(model_input, x, name='FC')


    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    a = time.time()
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    b = time.time()

    print(f"Total time: {b - a} seconds, Time per epoch ({epochs}): {(b - a) / epochs}")