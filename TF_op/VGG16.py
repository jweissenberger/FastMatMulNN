import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    model = tf.keras.applications.VGG16(include_top=True, weights=None, classes=1000, classifier_activation='softmax')

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    image = tf.image.decode_jpeg(tf.io.read_file('test_image.JPEG'))

    image = tf.keras.applications.vgg19.preprocess_input(image)

    image = tf.image.resize(image, (224, 224))
    image = tf.reshape(image, [1, 224, 224, 3])

    epochs = 2
    num_examples = 150

    y_train = tf.ones([num_examples])
    x_train = []
    for i in range(num_examples):
        x_train.append(image)

    x_train = tf.concat(x_train, 0)

    model.fit(x_train, y_train, batch_size=num_examples, epochs=epochs)
