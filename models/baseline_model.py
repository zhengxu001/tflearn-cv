"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script gives the network definition."""

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

def create_network(img_prep, img_aug, learning_rate):
    # """This function defines the network structure.

    # Args:
    #     img_prep: Preprocessing function that will be done to each input image.
    #     img_aug: Data augmentation function that will be done to each training input image.

    # Returns:
    #     The network."""

    # # Input shape will be [batch_size, height, width, channels].
    # network = input_data(shape=[None, 64, 64, 3],
    #                      data_preprocessing=img_prep,
    #                      data_augmentation=img_aug)

    # network = conv_2d(network, 64, 3, activation='relu', padding='SAME')
    # network = max_pool_2d(network, 2)
    # network = conv_2d(network, 32, 3, activation='relu', padding='SAME')
    # network = max_pool_2d(network, 2)
    # network = conv_2d(network, 16, 3, activation='relu', padding='SAME')
    # network = max_pool_2d(network, 2)
    # network = fully_connected(network, 1024, activation='relu')
    # network = dropout(network, 0.5)
    # network = fully_connected(network, 200, activation='softmax')
    # network = regression(network, optimizer='adam',
    #                      loss='categorical_crossentropy',
    #                      learning_rate=learning_rate)
    # return network




# def vgg_16(img_prep, img_aug, learning_rate)
  network = input_data(shape=[None, 64, 64, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

  network = conv_2d(network, 64, 3, activation='relu', padding='SAME')
  network = conv_2d(network, 64, 3, activation='relu', padding='SAME')
  network = max_pool_2d(network, 2)

  # (N, 28, 28, 64)
  network = conv_2d(network, 128, 3, activation='relu', padding='SAME')
  network = conv_2d(network, 128, 3, activation='relu', padding='SAME')
  network = max_pool_2d(network, 2)

  # (N, 14, 14, 128)
  network = conv_2d(network, 256, 3, activation='relu', padding='SAME')
  network = conv_2d(network, 256, 3, activation='relu', padding='SAME')
  network = conv_2d(network, 256, 3, activation='relu', padding='SAME')
  network = max_pool_2d(network, 2)

  # (N, 7, 7, 256)
  network = conv_2d(network, 256, 3, activation='relu', padding='SAME')
  network = conv_2d(network, 256, 3, activation='relu', padding='SAME')
  network = conv_2d(network, 256, 3, activation='relu', padding='SAME')

  # fc1: flatten -> fully connected layer
  # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
  network = fully_connected(network, 4096, activation='relu')
  network = dropout(network, 0.5)

  # fc2
  # (N, 4096) -> (N, 2048)
  network = fully_connected(network, 2048, activation='relu')
  network = dropout(network, 0.5)

  # softmax
  # (N, 2048) -> (N, 200)
  network = fully_connected(network, 200, activation='softmax')

  network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)

  return network