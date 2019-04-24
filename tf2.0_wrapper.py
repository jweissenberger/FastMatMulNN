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


class MLP(Layer):
    """Simple stack of Linear layers."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)

class MLPWithDropout(Layer):

    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training=None):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return self.linear_3(x)


if __name__ == '__main__':

    mlp = MLPWithDropout()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    # Create a training step function.

    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits = mlp(x)
            loss = loss_fn(y, logits)
            gradients = tape.gradient(loss, mlp.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))
        return loss

    # Prepare a dataset.
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype('float32') / 255, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    for step, (x, y) in enumerate(dataset):
        loss = train_on_batch(x, y)
        if step % 100 == 0:
            print(step, float(loss))
