from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

def resLayer(x, filters, m=False):
    network = conv_2d(x, filters, 3, activation='relu')
    network = batch_normalization(network)
    network = conv_2d(network, filters, 3, activation='relu')
    network = batch_normalization(network)
    if m == True:
        network = max_pool_2d(network, 2)
        return network
    return (x + network)

def create_network(img_prep, img_aug, learning_rate):
    network = input_data(shape=[None, 64, 64, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = resLayer(network, 64, m=True)
    network = resLayer(network, 64)

    network = resLayer(network, 128, m=True)
    network = resLayer(network, 128)

    network = resLayer(network, 256, m=True)
    network = resLayer(network, 256)

    network = resLayer(network, 512, m=True)
    network = resLayer(network, 512)

    network = fully_connected(network, 1024, activation='relu')
    network = batch_normalization(network)
    network = dropout(network, 0.5)
    network = fully_connected(network, 200, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network
