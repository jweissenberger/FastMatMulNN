import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.ops import array_ops, gen_math_ops, math_ops
import time
import argparse
from tensorflow.python.profiler import profiler_v2 as profiler
from openmpext import controlOMP



print(controlOMP(12))

# to change MKL's threads at runtime
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

print( "MKL num threads default: ", mkl_get_max_threads() )
mkl_set_num_threads(12)
print( "MKL num threads set to: ", mkl_get_max_threads() )

# To change TensorFlow's threads at runtime
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(1)



@tf.RegisterGradient("FastMatMul")
def _Fast_MatMul_grad(op, grad):
    #'''
    bt = array_ops.transpose(op.inputs[1])
    at = array_ops.transpose(op.inputs[0])
    grad_a = fast_mm_module.FastMatMul(a_matrix=grad, b_matrix=bt, epsilon=1e-2, steps=1, numthreads=12)
    grad_b = fast_mm_module.FastMatMul(a_matrix=at, b_matrix=grad, epsilon=1e-2, steps=1, numthreads=12)
    '''
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
    grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    #'''
    return grad_a, grad_b


class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32, mm_algorithm='regular'):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer='zeros',
                                 trainable=True)

        self.mm = mm_algorithm

        epsilon_values = {
            'bini322': 1e-2,

        }
        self.epsilon = epsilon_values.get(self.mm, 1e-2)

    def call(self, inputs):
        # the important lines:
        #'''
        if self.mm == 'regular':
            return tf.matmul(inputs, self.w) + self.b

        else:
            return fast_mm_module.FastMatMul(a_matrix=inputs, b_matrix=self.w, epsilon=self.epsilon, steps=1, numthreads=12) + self.b
        '''
        return tf.matmul(inputs, self.w) + self.b
        #'''


class MyModel(Model):
    def __init__(self, node, num_layers, matmul_algo):
        super(MyModel, self).__init__()

        self.num_layers = num_layers
        self.input_layer = Linear(input_dim=784, units=node, mm_algorithm='regular')
        self.output_layer = Linear(input_dim=node, units=10, mm_algorithm='regular')
        self.hidden = {}
        for i in range(num_layers):
            self.hidden[f'h{i}'] = Linear(input_dim=node, units=node, mm_algorithm=matmul_algo)

    def call(self, x):
        x = self.input_layer(x)
        x = tf.nn.relu(x)

        for i in range(self.num_layers):
            x = self.hidden[f'h{i}'](x)
            x = tf.nn.relu(x)

        x = self.output_layer(x)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mm", type=str)  # name of the matrix multiplication algorithm
    parser.add_argument("--logdir", type=str)

    args = parser.parse_args()
    batch_size = args.bs
    EPOCHS = args.epochs
    nodes = args.nodes
    layers = args.layers
    mm_algo = args.mm
    logdir = args.logdir

    if mm_algo != 'regular':
        fast_mm_module = tf.load_op_library(f'obj/{mm_algo}_mat_mul.so')

    tf.random.set_seed(100)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(60000, 784)
    x_train = x_train[:32768, :]
    #x_test = x_test.reshape(10000, 784)

    model = MyModel(node=nodes, num_layers=layers, matmul_algo=mm_algo)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    #test_loss = tf.keras.metrics.Mean(name='test_loss')
    #test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_accuracy_list = []
    train_loss_list = []
    #test_accuracy_list = []
    #test_loss_list = []

    overall_average_batch_time = 0

    profiler.warmup()
    profiler.start(logdir=logdir)

    total = 0
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        # test_loss.reset_states()
        # test_accuracy.reset_states()

        total_batch_time = 0
        batches = 0

        a = time.time()
        for batch in range(60000 // batch_size):
            x = time.time()
            train_step(x_train[batch * batch_size:(batch + 1) * batch_size],
                       y_train[batch * batch_size:(batch + 1) * batch_size])
            y = time.time()

            total_batch_time += y - x
            batches += 1

        # test_step(x_test, y_test)
        b = time.time()
        # print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Train Accuracy: {train_accuracy.result() * 100},'
        #       f'Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}')

        train_accuracy_list.append(train_accuracy.result())
        train_loss_list.append(train_loss.result())
        # test_accuracy_list.append(test_accuracy.result())
        # test_loss_list.append(test_loss.result())

        diff = b - a
        if epoch != 0:
            # don't want to count the first epoch so that it'll count as a warm up
            total += diff
            overall_average_batch_time += total_batch_time / batches
        #     print(f'Running Average Epoch time: {total / epoch}\n')
        # print(f'Time for Epoch:{diff}')
        # print(f'Average single batch time this epoch: {total_batch_time / batches}')

    # TODO output should be a single json object not multiple files with different information

    # write the performance lists to file
    with open(f'{mm_algo}_layers{layers}_nodes{nodes}_epochs{EPOCHS}_bs{batch_size}_accuracy_and_loss.txt', 'wt') as file:
        file.write('train_accuracy')
        for i in train_accuracy_list:
            file.write(f',{i}')

        file.write('\ntrain_loss')
        for i in train_loss_list:
            file.write(f',{i}')

        # file.write('\ntest_accuracy')
        # for i in test_accuracy_list:
        #     file.write(f',{i}')
        #
        # file.write('\ntest_loss')
        # for i in test_loss_list:
        #     file.write(f',{i}')

    # print(f'Average time per Batch: {overall_average_batch_time / (EPOCHS-1)}')
    # print(f'Average time per Epoch: {total / (EPOCHS-1)}')

    print(f"Algorithm: {mm_algo}")
    print(f"Total time: {total}")
    print(f"Matrix size: {batch_size}")  # this is assuming that bs and number of nodes is the same size


    profiler.stop()
    #python -u tensorboard_test.py --layers 2 --nodes 30 --epochs 5 --bs 64 --mm bini322
    #Run tensorboard --logdir logdir
