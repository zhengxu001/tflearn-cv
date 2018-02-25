import sys
import os
import argparse
import tflearn
import h5py
import itertools
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from datasets.tiny_imagenet import *
from models.alchNet11 import alchNet11
from models.alchNet19 import alchNet19
from models.vggNet import *
from models.resNet import *
from models.alexNet import *

from tflearn.data_utils import shuffle

def get_data(data_dir):
    train_file, val_file, conf_file = build_dataset_index(data_dir)
    print(conf_file)
    from tflearn.data_utils import image_preloader
    X_conf, Y_conf = image_preloader(conf_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)
    return X_conf, Y_conf

def create_net(model, img_prep, img_aug, learning_rate):
    if model == "alex":
        network = alexNet(img_prep, img_aug, learning_rate)
    elif model == "vgg":
        network = vggNet(img_prep, img_aug, learning_rate)
    elif model == "res":
        network = resNet(img_prep, img_aug, learning_rate)
    elif model == "alch11":
        network = alchNet11(img_prep, img_aug, learning_rate)
    elif model == "alch11_without_dropout":
        network = alchNet11(img_prep, img_aug, learning_rate, 0)
    else:
        network = alchNet19(img_prep, img_aug, learning_rate)
    return network

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


def get_class_names(data_dir):
    with open(os.path.join(data_dir, 'wnids.txt')) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    random_classes = random.sample(content,  5)
    return random_classes


data_dir = "data/tiny-imagenet-200"
target_path = "data/tiny-imagenet-200/cache/conf_image_paths.txt"
model_path = '/home/zen/tflearn-cv/output/vgg-NA-65/3052'
X_conf, Y_conf = get_data(data_dir)
img_prep = tflearn.data_preprocessing.ImagePreprocessing()
img_aug = tflearn.data_augmentation.ImageAugmentation()
learning_rate = 0.001
network = create_net("vgg", img_prep, img_aug, learning_rate)
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard')
model.load(model_path, weights_only=True)

with open(target_path, 'r') as f:
    images, labels = [], []
    for l in f.readlines():
        l = l.strip('\n').split()
        images.append(l[0])
        labels.append(int(l[1]))

# e = model.evaluate(images, labels)
y_pred = model.predict_label(images)
print(y_pred)
print(labels)
# predictions = []
# count = 0
# length = len(y_pred)
# for line in y_pred:
#   predictions.append(line[0])
#   count += 1
# print(count)
# print(predictions)

# cnf_matrix = confusion_matrix(Y_conf, y_pred)
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# plt.show()

# def main(name="alex-NA-60", num_epochs=60, aug_strategy="NA", model="alex"):
    # get_class_names()
    # X_test, Y_test = get_data(data_dir, model)
    # network = create_net(model, img_prep, img_aug, learning_rate)
    # model_path = '/home/zen/tflearn-cv/output/alex-NA-60/3105'
    # model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    # model.load(model_path, weights_only=True)
    # y_pred = model.predict_label(X_test)
    # cnf_matrix = confusion_matrix(Y_test, y_pred)

    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                   title='Normalized confusion matrix')
    # plt.show()

    # network = create_net(model, img_prep, img_aug, learning_rate)
    # network = regression(network, optimizer='adam',
    #                      loss='categorical_crossentropy',
    #                      learning_rate=learning_rate,
    #                      metric=tflearn.metrics.Top_k(k=1)
    #                     )
    # model_1 = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    # model_1.load(checkpoint_path + '1907', weights_only=True)
    # a = model_1.evaluate(X_test, Y_test, batch_size=batch_size)
    # print(a)

    # network = create_net(model, img_prep, img_aug, learning_rate)
    # network = regression(network, optimizer='adam',
    #                      loss='categorical_crossentropy',
    #                      learning_rate=learning_rate,
    #                      metric=tflearn.metrics.Top_k(k=5)
    #                     )
    # model_2 = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    # model_2.load(checkpoint_path + 'checkpoint')
    # b = model_2.evaluate(X_test, Y_test, batch_size=batch_size, weights_only=True)
    # import csv
    # with open('results.csv', 'wb') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',')
    #     spamwriter.writerow(name, aug, epoch, model, a, b)