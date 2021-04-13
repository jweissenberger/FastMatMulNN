"""VGG19 model for Keras.

Code copied from: https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/vgg19.py

# tensorflow 2.4.1

Reference:
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
      https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""

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


WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/'
                'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg19/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

layers = VersionAwareLayers()


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

    def call(self, inputs):
        print('\ninputs * weights')
        print('inputs shape:', inputs.shape)
        print('weights shape:', self.w.shape)
        output = fast_mm_module.FastMatMul(a_matrix=inputs, b_matrix=self.w, epsilon=self.epsilon, steps=1,
                                                    numthreads=num_threads) + self.b
        if self.activation == 'softmax':
            return tf.nn.softmax(output)
        else:
            return tf.nn.relu(output)


def VGG19(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
  """Instantiates the VGG19 architecture.
  Reference:
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
      https://arxiv.org/abs/1409.1556) (ICLR 2015)
  By default, it loads weights pre-trained on ImageNet. Check 'weights' for
  other options.
  This model can be built both with 'channels_first' data format
  (channels, height, width) or 'channels_last' data format
  (height, width, channels).
  The default input size for this model is 224x224.
  Note: each Keras Application expects a specific kind of input preprocessing.
  For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your
  inputs before passing them to the model.
  Arguments:
    include_top: whether to include the 3 fully-connected
      layers at the top of the network.
    weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)`
      (with `channels_last` data format)
      or `(3, 224, 224)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
  Returns:
    A `keras.Model` instance.
  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')
  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  # Block 1
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
          img_input)
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    print(x.shape)

    #x = layers.Dense(4096, activation='relu', name='fc1')(x)
    fast_layer1 = Fast_Linear(units=4096, input_dim=25088, activation='relu')
    x = fast_layer1(x)
    print(x.shape)

    #x = layers.Dense(4096, activation='relu', name='fc2')(x)
    fast_layer2 = Fast_Linear(units=4096, input_dim=4096, activation='relu')
    x = fast_layer2(x)
    print(x.shape)

    #imagenet_utils.validate_activation(classifier_activation, weights)

    #x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    fast_output_layer = Fast_Linear(units=classes, input_dim=4096, activation='softmax')
    x = fast_output_layer(x)
    print(x.shape)

  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='vgg19')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      weights_path = data_utils.get_file(
          'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='cbe5617147190e668d6c5d5026f83318')
    else:
      weights_path = data_utils.get_file(
          'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


if __name__ == '__main__':

    fast_mm_module = tf.load_op_library(f'obj/{mm_algo}_mat_mul.so')

    image = tf.image.decode_jpeg(tf.io.read_file('test_image.JPEG'))

    image = tf.keras.applications.vgg19.preprocess_input(image)

    image = tf.image.resize(image, (224, 224))
    image = tf.reshape(image, [1, 224, 224, 3])

    model = VGG19(include_top=True, weights=None, input_tensor=None, pooling=None)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    epochs = 2
    num_examples = 1024

    y_train = tf.ones([num_examples])
    x_train = []

    x_train = tf.concat(x_train, 0)

    a = time.time()
    model.fit(x_train, y_train, batch_size=1024, epochs=epochs)
    b = time.time()

    print(f"Total train time: {b-a} seconds\nTime per epoch: {(b-a)/epochs}")
