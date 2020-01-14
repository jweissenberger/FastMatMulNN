import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops

classic_mm_module = tf.load_op_library('./classic_mat_mul.so')

@tf.RegisterGradient("ClassicMatMul")
def _ClassicMatMulGrad(op, grad):
    t_a = op.get_attr("transpose_a")  #https://github.com/tensorflow/tensorflow/blob/b00e936e42f3370256bc7d17d4f7685ae7cd750f/tensorflow/core/kernels/matmul_op.cc#L272
    t_b = op.get_attr("transpose_b")
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    if not t_a and not t_b:
        grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    elif not t_a and t_b:
        grad_a = gen_math_ops.mat_mul(grad, b)
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
    elif t_a and not t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, grad)
    elif t_a and t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True)
    return grad_a, grad_b

class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        #return tf.matmul(inputs, self.w) + self.b
        return classic_mm_module.ClassicMatMul(a_matrix=inputs, b_matrix=self.w) + self.b


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.d1 = Linear(input_dim=784, units=100)
        self.d2 = Linear(input_dim=100, units=300)
        self.d3 = Linear(input_dim=300, units=100)
        self.out = Linear(input_dim=100, units=10)

    def call(self, x):
        x = self.d1(x)
        x = tf.nn.relu(x)
        x = self.d2(x)
        x = tf.nn.relu(x)
        x = self.d3(x)
        x = tf.nn.relu(x)
        x = self.out(x)
        x = tf.nn.softmax(x)
        return x


@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


if __name__ == '__main__':
    #classic_mm_module = tf.load_op_library('./classic_mat_mul.so')

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    batch_size = 64

    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch in range(60000//batch_size):
            train_step(x_train[batch*batch_size:(batch+1)*batch_size], y_train[batch*batch_size:(batch+1)*batch_size])

        test_step(x_test, y_test)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
