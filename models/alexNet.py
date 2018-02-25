from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def alexNet(img_prep, img_aug, learning_rate):

    network = input_data(shape=[None, 64, 64, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)

    network = conv_2d(network, 96, 9, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 200, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)

    return network
