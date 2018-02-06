from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.activations import relu
from tflearn.layers.normalization import batch_normalization

def resLayer(x, filters, stride=1):
    network = conv_2d(x, filters, 3, activation=None, stride=stride)
    network = batch_normalization(network)
    network = relu(network)
    network = conv_2d(network, filters, 3, activation=None)
    network = batch_normalization(network)
    if stride != 1:
      print("Projecting identity mapping to correct size")
      x = max_pool_2d(x, 2)
      x = conv_2d(x, filters, 1)

    network = x + network
    network = relu(network)

    return network


def create_network(img_prep, img_aug, learning_rate):
    network = input_data(shape=[None, 56, 56, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    network = conv_2d(network, 64, 3, activation='relu')
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)

    network = resLayer(network, 64)
    network = resLayer(network, 64)
    network = max_pool_2d(network, 2)

    network = resLayer(network, 128)
    network = resLayer(network, 128)
    network = max_pool_2d(network, 2)

    network = resLayer(network, 256)
    network = resLayer(network, 256)
    network = max_pool_2d(network, 2)

    network = resLayer(network, 512)
    network = resLayer(network, 512)
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 1024, activation='relu')
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    network = dropout(network, 0.5)
    network = fully_connected(network, 200, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network
