from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, residual_block, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.activations import relu

def resNet(img_prep, img_aug, learning_rate):
    network = input_data(shape=[None, 64, 64, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    network = conv_2d(network, 64, 3, regularizer='L2', weight_decay=0.0001)
    network = residual_block(network, 2, 64)
    network = residual_block(network, 1, 128, downsample=True)
    network = residual_block(network, 1, 128)
    network = residual_block(network, 1, 256, downsample=True)
    network = residual_block(network, 1, 256)
    network = residual_block(network, 1, 512, downsample=True)
    network = residual_block(network, 1, 512)
    network = batch_normalization(network)
    network = relu(network)
    network = global_avg_pool(network)
    network = fully_connected(network, 200, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network
