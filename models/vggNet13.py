from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def vggNet13(img_prep, img_aug, learning_rate):
    network = input_data(shape=[None, 64, 64, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
    
    network = conv_2d(network, 64, 3, activation='relu', scope='conv1_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv1_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')
    network = conv_2d(network, 128, 3, activation='relu', scope='conv2_1')
    network = conv_2d(network, 128, 3, activation='relu', scope='conv2_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')
    network = conv_2d(network, 256, 3, activation='relu', scope='conv3_1')
    network = conv_2d(network, 256, 3, activation='relu', scope='conv3_2')
    network = conv_2d(network, 256, 3, activation='relu', scope='conv3_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv4_1')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv4_2')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv4_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')
    network = fully_connected(network, 1024, activation='relu', scope='fc6')
    network = dropout(network, 0.5, name='dropout1')
    network = fully_connected(network, 1024, activation='relu', scope='fc7')
    network = dropout(network, 0.5, name='dropout2')
    network = fully_connected(network, 200, activation='softmax', scope='fc8')

    network = regression(network, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=learning_rate)

    return network