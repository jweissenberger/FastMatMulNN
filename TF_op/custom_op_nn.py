import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.ops import array_ops
import time

# to change MKL's threads at runtime
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

fast_mm_module = tf.load_op_library('obj/schonhage_mat_mul.so')

print( mkl_get_max_threads() )
mkl_set_num_threads(1)
print( mkl_get_max_threads() )


@tf.RegisterGradient("FastMatMul")
def _Fast_MatMul_grad(op, grad):
    # here I am using our classic matmul to pass the gradients back in backprop, I could use classic mat mul here as well
    # must use ops in this, not normal tensorflow calls
    bt = array_ops.transpose(op.inputs[1])
    at = array_ops.transpose(op.inputs[0])
    grad_a = fast_mm_module.FastMatMul(a_matrix=grad, b_matrix=bt, epsilon=1e-2, steps=1)
    grad_b = fast_mm_module.FastMatMul(a_matrix=at, b_matrix=grad, epsilon=1e-2, steps=1)
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
        # this is the multiplication, can use normal tensorflow code here as well
        return fast_mm_module.FastMatMul(a_matrix=inputs, b_matrix=self.w, epsilon=1e-2, steps=1) + self.b


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        layer1 = 800
        layer2 = 700
        layer3 = 100

        self.d1 = Linear(input_dim=784, units=layer1)
        self.d2 = Linear(input_dim=layer1, units=layer2)
        self.d3 = Linear(input_dim=layer2, units=layer3)
        self.out = Linear(input_dim=layer3, units=10)

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

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    batch_size = 64

    train_times = 0
    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    EPOCHS = 100
    total = 0
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        a = time.time()
        for batch in range(60000//batch_size):
            train_step(x_train[batch*batch_size:(batch+1)*batch_size], y_train[batch*batch_size:(batch+1)*batch_size])

        test_step(x_test, y_test)
        b = time.time()
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        diff = b-a
        total += diff
        print(f'Time for Epoch:{diff}')
        if epoch != 0:
            print(f'Running Average: {total / epoch}\n')
        train_times += total
    print(f'Average time per Epoch:{total / EPOCHS}')
