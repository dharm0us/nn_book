#tested with python2.7
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

def n3():
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    #iter1
    #net = network3.Network([ network3.FullyConnectedLayer(n_in=784, n_out=100), network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    #iter2
    net =  network3.Network([ network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        network3.FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
        network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 10, mini_batch_size, 0.1, validation_data, test_data)

n3()
