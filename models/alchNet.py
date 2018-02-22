from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

def alchNet(img_prep, img_aug, learning_rate):
    network = input_data(shape=[None, 64, 64, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 5, activation='relu')
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu')
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    network = fully_connected(network, 1024, activation='relu')
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    network = dropout(network, 0.5)
    network = fully_connected(network, 200, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network
