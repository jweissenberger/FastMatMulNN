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


# output to second FC
@tf.RegisterGradient("FastMatMul552")
def first_Fast_MatMul_grad(op, grad):
    bt = array_ops.transpose(op.inputs[1])
    at = array_ops.transpose(op.inputs[0])
    grad_a = fast_mm_442(a_matrix=grad, b_matrix=bt, epsilon=1e-2, steps=1, numthreads=num_threads)
    grad_b = fast_mm_525(a_matrix=at, b_matrix=grad, epsilon=1e-2, steps=1, numthreads=num_threads)
    return grad_a, grad_b


# second FC to first
@tf.RegisterGradient("FastMatMul424")
def second_Fast_MatMul_grad(op, grad):
    bt = array_ops.transpose(op.inputs[1])
    at = array_ops.transpose(op.inputs[0])
    grad_a = fast_mm_424(a_matrix=grad, b_matrix=bt, epsilon=1e-2, steps=1, numthreads=num_threads)
    grad_b = fast_mm_424(a_matrix=at, b_matrix=grad, epsilon=1e-2, steps=1, numthreads=num_threads)
    return grad_a, grad_b


# second FC to first
@tf.RegisterGradient("FastMatMul244")
def thrid_Fast_MatMul_grad(op, grad):
    bt = array_ops.transpose(op.inputs[1])
    at = array_ops.transpose(op.inputs[0])
    grad_a = fast_mm_442(a_matrix=grad, b_matrix=bt, epsilon=1e-2, steps=1, numthreads=num_threads)
    grad_b = fast_mm_444(a_matrix=at, b_matrix=grad, epsilon=1e-2, steps=1, numthreads=num_threads)
    return grad_a, grad_b


class Fast_Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, activation='relu', trainable=True, mm_module=None, mm_algo='smirnov442'):
        super(Fast_Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=trainable,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=trainable
        )

        self.epsilon = epsilon_values.get(mm_algo, 1e-2)

        self.activation = activation

        self.fast_mm_module = mm_module

    def call(self, inputs):
        output = self.fast_mm_module(a_matrix=inputs, b_matrix=self.w, epsilon=self.epsilon, steps=1,
                                                    numthreads=num_threads) + self.b
        if self.activation == 'softmax':
            return tf.nn.softmax(output)
        else:
            return tf.nn.relu(output)


num_threads = 6
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

    fast_mm_242 = tf.load_op_library('obj/smirnov242_mat_mul.so')
    fast_mm_242 = fast_mm_242.FastMatMul242
    fast_mm_442 = tf.load_op_library('obj/smirnov442_mat_mul.so')
    fast_mm_442 = fast_mm_442.FastMatMul442
    fast_mm_525 = tf.load_op_library('obj/smirnov525_mat_mul.so')
    fast_mm_525 = fast_mm_525.FastMatMul525
    fast_mm_424 = tf.load_op_library('obj/smirnov424_mat_mul.so')
    fast_mm_424 = fast_mm_424.FastMatMul424
    fast_mm_444 = tf.load_op_library('obj/smirnov444_mat_mul.so')
    fast_mm_444 = fast_mm_444.FastMatMul444
    fast_mm_244 = tf.load_op_library('obj/smirnov244_mat_mul.so')
    fast_mm_244 = fast_mm_244.FastMatMul244
    fast_mm_552 = tf.load_op_library('obj/smirnov552_mat_mul.so')
    fast_mm_552 = fast_mm_552.FastMatMul552



    epochs = 3
    batch_size = 1024

    y_train = tf.random.uniform(shape=[batch_size])
    x_train = tf.random.uniform(shape=[batch_size, 2])

    model_input = layers.Input(shape=2)

    #fast_layer0 = Fast_Linear(units=25088, input_dim=2, activation='relu', mm_module=fast_mm_442)
    #x = fast_layer0(model_input)
    x = layers.Dense(25088, activation='relu', name='fc0')(model_input)

    # x = layers.Dense(4096, activation='relu', name='fc1')(x)
    fast_layer1 = Fast_Linear(units=4096, input_dim=25088, activation='relu', mm_module=fast_mm_244)
    x = fast_layer1(x)

    # x = layers.Dense(4096, activation='relu', name='fc2')(x)
    fast_layer2 = Fast_Linear(units=4096, input_dim=4096, activation='relu', mm_module=fast_mm_424)
    x = fast_layer2(x)

    # imagenet_utils.validate_activation('softmax', weights)

    # x = layers.Dense(1000, activation='softmax', name='predictions')(x)
    fast_output_layer = Fast_Linear(units=1000, input_dim=4096, activation='softmax', mm_module=fast_mm_244)
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
              )

    b = time.time()

    print(f"\n\nTotal time: {b - a} seconds, Time per epoch ({epochs}): {(b - a) / epochs}")
    print(f"Batch size: {batch_size}, mm algo: 442")

    #Total time: 6.033765554428101 seconds, Time per epoch (2): 3.0168827772140503