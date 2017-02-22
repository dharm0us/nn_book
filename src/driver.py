#tested with python2.7
import time

import network
import mnist_loader
import network2
import network3

def n():
    #net = network.Network([784, 30, 10])
    net = network.Network([784, 10])
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def n2():
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.large_weight_initializer()
    ec,ea,tc,ta = net.SGD(training_data, 2, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy = True,monitor_training_cost=True)
    print(ec)
    print(ea)
    print(tc)
    print(ta)

def n3_noconv():
    #74.6 seconds - GPU false
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = network3.Network([ network3.FullyConnectedLayer(n_in=784, n_out=100), network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 10, mini_batch_size, 0.1, validation_data, test_data)

def n3_one_conv():
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net =  network3.Network([ network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        network3.FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
        network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 10, mini_batch_size, 0.1, validation_data, test_data)

def n3_twoconv():
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = network3.Network([
        network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        network3.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        network3.FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100),
        network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 10, mini_batch_size, 0.1, validation_data, test_data)

def n3_relu_regularization():
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = network3.Network([
        network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        network3.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        network3.FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100, activation_fn=network3.ReLU),
        network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

def n3_dropout_and_expanded_data():
    #we have applied dropout to only fully connected layers and not the convolutional layers, as those layers handle the
    #overfitting better
    training_data, validation_data, test_data = network3.load_data_shared()
    expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
    mini_batch_size = 10
    net = network3.Network([
        network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        network3.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        network3.FullyConnectedLayer(
            n_in=40 * 4 * 4, n_out=1000, activation_fn=network3.ReLU, p_dropout=0.5),
        network3.FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=network3.ReLU, p_dropout=0.5),
        network3.SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
    net.SGD(expanded_training_data, 40, mini_batch_size, 0.03,
                 validation_data, test_data)

def timedRun(methodToRun):
    start = time.time()
    result = methodToRun()
    end = time.time()
    diff = end - start
    print("Method: " + str(methodToRun) + "Time Taken :" + str(diff))
    print(result)

#timedRun(n3_noconv)
#timedRun(n3_twoconv)
timedRun(n3_relu_regularization)
