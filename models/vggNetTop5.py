import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def vggNetTop5(img_prep, img_aug, learning_rate):
    network = input_data(shape=[None, 64, 64, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5, name='dropout1')
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5, name='dropout2')
    network = fully_connected(network, 200, activation='softmax')

    network = regression(network, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=learning_rate,
                        metric=tflearn.metrics.Top_k(k=5)
                        )
    return network