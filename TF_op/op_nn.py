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
        return classic_mm_module(inputs, self.w) + self.b


class Network(Layer):

    def __init__(self):
        super(Network, self).__init__()
        self.linear_1 = Linear(500)
        self.linear_2 = Linear(300)
        self.linear_3 = Linear(200)
        self.linear_4 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2
        x = tf.nn.relu(x)
        x = self.linear_3
        x = tf.nn.relu(x)
        return self.linear_4(x)


if __name__ == '__main__':
    classic_mm_module = tf.load_op_library('./classic_mat_mul.so')

    batch_size = 128
    optimizer = tf.keras.optimizers.Adam()
    network = Network()
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype('float32') / 255, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)


    # input_shape=(28, 28)

    @tf.function
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits = network(x)
            loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=logits, from_logits=True)
            gradients = tape.gradient(loss, network.trainable_weights)
        optimizer.apply_gradients(zip(gradients, network.trainable_weights))
        return loss


    for step, (x, y) in enumerate(dataset):
        loss = train_on_batch(x, y)
        if step % 100 == 0:
            print(step, float(loss))