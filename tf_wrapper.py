import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from numpy import linalg as la


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


def bini(A, B, steps, e=1e-8):

    #Check Dimensions
    (m, n) = A.get_shape().as_list()
    #rn assuming that m is bigger than n, nn and p
    (nn, p) = B.get_shape().as_list()
    if n != nn: raise ValueError("incompatible dimensions")

    #pre-allocate output matrix
    C = tf.zeros([m,p])

    """
    This is the notation I use from Bini's 1980 paper

    |A1, A4|  |B1, B2|  =  |C1, C2|
    |A2, A5|  |B3, B4|     |C3, C4|
    |A3, A6|               |C5, C6|
    """

    # Base case
    if steps == 0 or m == 1 or n == 1 or p == 1:
        C = tf.matmul(A,B)
        return C

    # Static peeling
    if (3**steps > m) or (2**steps > n) or (2**steps > p):
        raise ValueError("Too many steps/ too small matricies for static peeling")

    if (m % 3**steps) != 0:
        extra_rows = m % 3**steps

        #C[:m-extra_rows, :] =
        Cmat = bini(A[:m-extra_rows, :], B, steps, e)
        #C[m-extra_rows:, :] =

        # need to expand dims if slice of A is a vector, and expand dims if it is
        A_slice = A[m-extra_rows:, :]
        (slice_len, guy) = A_slice.get_shape().as_list()

        Crow = tf.matmul(A_slice, B)
        C = tf.concat([Cmat, Crow], 0)
        return C

    if (n % 2**steps) != 0:
        extra_cols = n % (2**steps)

        Cmat = bini(A[:, :n-extra_cols], B[:n-extra_cols,:], steps, e)

        A_slice = A[:, n-extra_cols:]
        B_slice = B[n-extra_cols:, :]
        (_, slice_len) = A_slice.get_shape().as_list()

        Ccol = tf.matmul(A_slice, B_slice)

        C = tf.add(Cmat, Ccol)
        return C

    if (p % 2**steps) != 0:
        multiP = p//(2**steps)  # multiplier to find how large to make the bini matrix
        extra_cols = p % (2**steps)

        Cmat = bini(A, B[:, :p-extra_cols], steps, e)

        B_slice = B[:, p-extra_cols:]
        (_, slice_len) = B_slice.get_shape().as_list()

        Ccol = tf.matmul(A, B_slice)

        C = tf.concat([Cmat, Ccol], 1)
        return C

    """
    Dynamic peeling causes issues because the ideal epsilon value is determined by 
    the shape of the matrix and in dynamic peeling, the shape of the matrix
    is changed every recursive step which results in dimensions with a different 
    ideal epsilon value
    
    #Dynamic peeling
    if m % 3 == 1:
        C[:m-1, :] = bini(A[:m-1,:],B, steps, e)
        C[m-1,:] = A[m-1,:]@B
        return C
    if m % 3 == 2:
        C[:m-2, :] = bini(A[:m-2,:],B, steps, e)
        C[m-2:,:] = A[m-2:,:]@B
        return C
    if n % 2 == 1:
        C = bini(A[:, :n-1], B[:n-1,:], steps, e)
        C = C + np.outer(A[:,n-1],B[n-1,:])
        return C
    if p % 2 == 1:
        C[:, :p-1] = bini(A, B[:,:p-1], steps, e)
        C[:,p-1] = A@B[:,p-1]
        return C
    """


    # split up the matricies once rows of A are divisible by 3
    # and cols of A and rows and cols of are divisible by 2
    m2 = int(m/3) #first third of the rows of A
    m3 = m2*2     #second third of the rows of A
    n2 = int(n/2) #half of the cols of A
    p2 = int(p/2) #half of the cols of B
    #nn2 = int(nn/2) # half of the rows of B

    A1 = A[:m2, :n2]
    A2 = A[m2:m3, :n2]
    A3 = A[m3:, :n2]
    A4 = A[:m2, n2:]
    A5 = A[m2:m3, n2:]
    A6 = A[m3:, n2:]

    B1 = B[:n2, :p2]
    B2 = B[:n2, p2:]
    B3 = B[n2:, :p2]
    B4 = B[n2:, p2:]

    # conquer

    # check if TF has a special fun for scalar mul
    M1  = bini(tf.add(A1, A5)                  , tf.add(tf.scalar_mul(e, B1), B4)     , steps-1, e)
    M2  = bini(A5                              , tf.subtract(-B3, B4)                 , steps-1, e)
    M3  = bini(A1                              , B4                                   , steps-1, e)
    M4  = bini(tf.add(tf.scalar_mul(e,A4), A5) , tf.add(tf.scalar_mul(-e, B1), B3)    , steps-1, e)
    M5  = bini(tf.add(A1, tf.scalar_mul(e, A4)), tf.add(tf.scalar_mul(e, B2), B4)     , steps-1, e)
    M6  = bini(tf.add(A2, A6)                  , tf.add(B1, tf.scalar_mul(e, B4))     , steps-1, e)
    M7  = bini(A2                              , tf.subtract(-B1, B2)                 , steps-1, e)
    M8  = bini(A6                              , B1                                   , steps-1, e)
    M9  = bini(tf.add(A2, tf.scalar_mul(e, A3)), tf.subtract(B2, tf.scalar_mul(e, B4)), steps-1, e)
    M10 = bini(tf.add(tf.scalar_mul(e, A3), A6), tf.add(B1, tf.scalar_mul(e, B3))     , steps-1, e)

    # nation building
    # gonna have to con cat these, cant use indexing to put this together
    C1 = tf.scalar_mul((1/e), tf.add(M1, tf.subtract(tf.add(M2, M4), M3)))    #C[:m2, :p2]
    C2 = tf.scalar_mul((1/e), tf.add(-M3, M5))                                #C[:m2, p2:]
    C3 = tf.add(M4, tf.subtract(M6, M10))                                     #C[m2:m3, :p2] error from bini paper -M10 from +M10
    C4 = tf.add(tf.subtract(M1, M5), M9)                                      #C[m2:m3, p2:] error from bini paper -M5 from +M5
    C5 = tf.scalar_mul((1/e), tf.add(-M8,M10))                                #C[m3:, :p2]
    C6 = tf.scalar_mul((1/e), (tf.add(M6, tf.subtract(tf.add(M7, M9), M8))))  #C[m3:, p2:]

    # need to put all of the above pieces together
    C13 = tf.concat([C1, C3], 0)
    C135 = tf.concat([C13, C5], 0)
    C24 = tf.concat([C2, C4], 0)
    C246 = tf.concat([C24, C6], 0)
    C = tf.concat([C135, C246], 1)

    return C


def calculate_e(steps):
    # should be 26 if its double, 52 if its floating point
    e = (2**-26)**(1/(1+steps))
    return e


def neuron_layer(X, n_neurons, name, num_recursive_steps, fastmm='s', activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")

        # for Strassen's fast matrix multiply
        if fastmm == 's':
            Z = strass(X, W, num_recursive_steps) + b

        # for Bini's fast matrix mulitply
        if fastmm == 'b':
            # check what precision they use in backend
            #calculate_e(num_recursive_steps)
            Z = bini(X, W, steps=num_recursive_steps, e=0.1) + b

        if activation is not None:
            return activation(Z)
        else:
            return Z


if __name__ == '__main__':

    batch_size = 100
    seed = 25
    learning_rate = 0.01
    n_epochs = 50
    num_recur_steps = 1
    num_neural_nets = 20

    # should change the name of this every time you run a different time
    avg_epoch_test_accuracy = np.zeros(n_epochs)
    epoch_test_name = 'strass_1step_50eps_20nets_test'
    avg_epoch_train_accuracy = np.zeros(n_epochs)
    epoch_train_name = 'strass_1step_50eps_20nets_train'

    n_inputs = 28*28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 300
    n_outputs = 10

    epoch_acc97s = 0  # will have the sum of the epochs at which 97% accuracy was reached, will be used to find average
    final_test_accs = 0  # will have sum of final test acc, used to calculate average

    mnist = input_data.read_data_sets("/tmp/data/")
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    for q in range(num_neural_nets):
        tf.reset_default_graph()  # clears the computational graph

        tf.set_random_seed(seed+q)  # updates the seed so that each net is different

        X = tf.placeholder(tf.float32, shape=(batch_size, n_inputs), name="X")
        y = tf.placeholder(tf.int64, shape=(batch_size), name="y")

        with tf.name_scope("dnn"):
            hidden1 = neuron_layer(X, n_hidden1, num_recursive_steps=num_recur_steps, fastmm='b',
                                   name="hidden1", activation=tf.nn.relu)
            hidden2 = neuron_layer(hidden1, n_hidden2, num_recursive_steps=num_recur_steps, fastmm='b',
                                   name="hidden2", activation=tf.nn.relu)
            logits = neuron_layer(hidden2, n_outputs, num_recursive_steps=num_recur_steps, fastmm='b',
                                  name="outputs")

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

                #acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

                num_batches_in_test = mnist.test.num_examples // batch_size
                num_batches_in_train = mnist.train.num_examples // batch_size
                acc_test = 0
                acc_train = 0

                # have to use a loop to calc the acc because X and y are placeholders and cannot change sizes
                # which is needed for Bini and Strassen
                for j in range(num_batches_in_test):
                    acc_test += accuracy.eval(feed_dict={X: mnist.test.images[j*batch_size:batch_size*(j+1)],
                                                         y: mnist.test.labels[j*batch_size:batch_size*(j+1)]})

                for j in range(num_batches_in_train):
                    acc_train += accuracy.eval(feed_dict={X: mnist.train.images[j*batch_size:batch_size*(j+1)],
                                                          y: mnist.train.labels[j*batch_size:batch_size*(j+1)]})

                acc_test /= num_batches_in_test
                acc_train /= num_batches_in_train

                final_test_acc = acc_test  # keeps track of the test accuracy on the last epoch

                # adds the current test and train accuracy to that position so that the average can be calculated
                avg_epoch_train_accuracy[epoch] += acc_train
                avg_epoch_test_accuracy[epoch] += acc_test

                # keeps track of which epoch 97% test accuracy was reached, giving perspective on how quickly the
                # nets learn
                if acc_test >= 0.97 and epoch_w97acc < 2:
                    epoch_w97acc = epoch

                print(epoch, "Train accuracy:", acc_train, "Test accuracy", acc_test)

            print('\nNetwork:', q)
            print('Final test accuracy:', final_test_acc, '\nEpoch where 97% test accuracy was reached:',
                  epoch_w97acc, end='\n\n')

            final_test_accs += final_test_acc
            epoch_acc97s += epoch_w97acc

    # calculate average final test accuracy
    avg_final_test_acc = final_test_accs/num_neural_nets
    avg_97epoch = epoch_acc97s / num_neural_nets

    print('\nAvg final test accuracy:', avg_final_test_acc, '\nAvg epoch where 97% test accuracy was reached:',
          avg_97epoch)

    # save the average test and train accuracy per epoch so that they can be plotted in a notebook
    avg_epoch_test_accuracy /= num_neural_nets
    avg_epoch_train_accuracy /= num_neural_nets
    np.save(epoch_test_name, avg_epoch_test_accuracy)
    np.save(epoch_train_name, avg_epoch_train_accuracy)

