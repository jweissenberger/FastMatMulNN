import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def strass(A, B, steps):

    #Check Dimensions
    # tensor.get_shape().as_list()
    (m, n) = A.get_shape().as_list()
    (nn, p) = B.get_shape().as_list()

    #old code case m, n, nn, and p as ints

    if n != nn: raise ValueError("incompatible dimensions")
    C = tf.zeros([m,p])

    #Base case
    if steps == 0 or m ==1 or n ==1 or p == 1:
        C = tf.matmul(A,B)
        return C

    #Dynamic peeling
    # *****************
    if m % 2 == 1:
        #C[:m-1, :]
        Cmat= strass(A[:m-1,:],B, steps)
        #C[m-1,:], need to expand the dims b/c tf.matmul doesn't work for 1D vectors
        Crow = tf.matmul(tf.expand_dims(A[m-1,:],0),B)
        return tf.concat([Cmat, Crow], 0)
    if n % 2 == 1:
        Cmat = strass(A[:, :n-1], B[:n-1,:], steps)
        C = tf.add(Cmat,  tf.matmul(tf.expand_dims(A[:,n-1],1),tf.expand_dims(B[n-1,:],0)))
        return C
    if p % 2 == 1:
        #C[:, :p-1]
        Cmat = strass(A, B[:,:p-1], steps)
        #C[:,p-1]
        Ccol = tf.matmul(A,tf.expand_dims(B[:,p-1],1))
        return tf.concat([Cmat, Ccol], 1)

    # divide when m, n and p are all even
    m2 = int(m/2)
    n2 = int(n/2)
    p2 = int(p/2)
    A11 = A[:m2,:n2]
    A12 = A[:m2,n2:]
    A21 = A[m2:,:n2]
    A22 = A[m2:,n2:]
    B11 = B[:n2,:p2]
    B12 = B[:n2,p2:]
    B21 = B[n2:,:p2]
    B22 = B[n2:,p2:]

    # conquer
    M1 = strass(A11, tf.subtract(B12,B22)   ,steps-1)
    M2 = strass(tf.add(A11,A12), B22   ,steps-1)
    M3 = strass(tf.add(A21,A22),B11    ,steps-1)
    M4 = strass(A22    ,tf.subtract(B21,B11),steps-1)
    M5 = strass(tf.add(A11, A22), tf.add(B11, B22),steps-1)
    M6 = strass( tf.subtract(A12,A22), tf.add(B21,B22),steps-1)
    M7 = strass(tf.subtract(A11,A21), tf.add(B11, B12),steps-1)

    # conquer
    #C[:m2,:p2]
    C11 = tf.add(tf.subtract(tf.add(M5, M4), M2), M6)
    #C[:m2,p2:]
    C12 = tf.add(M1, M2)
    #C[m2:,:p2]
    C21 = tf.add(M3,M4)
    #C[m2:,p2:]
    C22 = tf.subtract(tf.subtract(tf.add(M1,M5), M3), M7)

    # nation building
    C1 = tf.concat([C11, C12], 1)
    C2 = tf.concat([C21,C22], 1)
    C = tf.concat([C1,C2], 0)

    return C


def neuron_layer(X, n_neurons, name, num_recursive_steps, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")

        Z = strass(X, W, num_recursive_steps) + b

        if activation is not None:
            return activation(Z)
        else:
            return Z


if __name__ == '__main__':

    batch_size = 100
    seed = 18
    learning_rate = 0.01
    n_epochs = 50
    num_recur_steps = 1
    num_neural_nets = 10

    n_inputs = 28*28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 300
    n_outputs = 10

    epoch_acc97s = []
    final_test_accs = []

    mnist = input_data.read_data_sets("/tmp/data/")
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    for q in range(num_neural_nets):
        tf.reset_default_graph()

        tf.set_random_seed(seed+q)

        X = tf.placeholder(tf.float32, shape=(batch_size, n_inputs), name="X")
        y = tf.placeholder(tf.int64, shape=(batch_size), name="y")

        with tf.name_scope("dnn"):
            hidden1 = neuron_layer(X, n_hidden1, num_recursive_steps=num_recur_steps, name="hidden1",
                                   activation=tf.nn.relu)
            hidden2 = neuron_layer(hidden1, n_hidden2, num_recursive_steps=num_recur_steps, name="hidden2",
                                   activation=tf.nn.relu)
            logits = neuron_layer(hidden2, n_outputs, num_recursive_steps=num_recur_steps, name="outputs")

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()
            epoch_w97acc = 0
            final_test_acc = 0

            for epoch in range(n_epochs):
                for iteration in range(mnist.train.num_examples // batch_size):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

                num_batches_in_test = mnist.test.num_examples // batch_size
                acc_test = 0
                for j in range(num_batches_in_test):
                    acc_test += accuracy.eval(feed_dict={X: mnist.test.images[j*batch_size:batch_size*(j+1)],
                                                         y: mnist.test.labels[j*batch_size:batch_size*(j+1)]})
                acc_test /= num_batches_in_test

                final_test_acc = acc_test

                if acc_test >= 0.97 and epoch_w97acc < 2:
                    epoch_w97acc = epoch

                print(epoch, "Train accuracy:", acc_train, "Test accuracy", acc_test)

            print('\nNetwork:', q)
            print('Final test accuracy:', final_test_acc, '\nEpoch where 97% test accuracy was reached:',
                  epoch_w97acc, end='\n\n')

            final_test_accs.append(final_test_acc)
            epoch_acc97s.append(epoch_w97acc)

    avg_final_test_acc = 0
    for j in final_test_accs:
        avg_final_test_acc += j

    avg_final_test_acc /= len(final_test_accs)

    avg_97epoch = 0
    for j in epoch_acc97s:
        avg_97epoch += j

    avg_97epoch /= len(epoch_acc97s)

    print('\nAvg final test accuracy:', avg_final_test_acc, '\nAvg epoch where 97% test accuracy was reached:',
          avg_97epoch)
