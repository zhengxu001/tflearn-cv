import sys
import os
import argparse
import tflearn
import h5py

sys.path.insert(0, os.path.dirname(__file__))

from datasets.tiny_imagenet import *
from models.alchNet11 import alchNet11
from models.alchNet19 import alchNet19
from models.vggNet import *
from models.vggNetTop5 import *
from models.vggNet13 import *
from models.vggNet16 import *
from models.resNet import *
from models.alexNet import *
from models.alchNetEnhance import *

from tflearn.data_utils import shuffle

def get_data(data_dir, model):
    train_file, val_file, conf_file = build_dataset_index(data_dir)

    # if not os.path.exists('hdf5'):
    #     os.makedirs('hdf5')
    # if not os.path.exists('hdf5/tiny-imagenet_train.h5'):
    #     from tflearn.data_utils import build_hdf5_image_dataset
    #     print('Creating hdf5 train dataset.')
    #     build_hdf5_image_dataset(train_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_train.h5', categorical_labels=True, normalize=True)

    # if not os.path.exists('hdf5/tiny-imagenet_val.h5'):
    #     from tflearn.data_utils import build_hdf5_image_dataset
    #     print(' Creating hdf5 val dataset.')
    #     build_hdf5_image_dataset(val_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_val.h5', categorical_labels=True, normalize=True)

    # # Load training data from hdf5 dataset.
    # h5f = h5py.File('hdf5/tiny-imagenet_train.h5', 'r')
    # X = h5f['X']
    # Y = h5f['Y']

    # # Load validation data.
    # h5f = h5py.File('hdf5/tiny-imagenet_val.h5', 'r')
    # X_test = h5f['X']
    # Y_test = h5f['Y']

    from tflearn.data_utils import image_preloader
    X, Y = image_preloader(train_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)
    X_test, Y_test = image_preloader(val_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)
    X, Y = shuffle(X, Y)
    return X, Y, X_test, Y_test

def set_data_augmentation(model, aug_strategy):
    if aug_strategy!="NA":
        print("Use Flip and Random Crop as data augmentation\n")
        img_aug = tflearn.data_augmentation.ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_flip_updown()
        img_aug.add_random_rotation (max_angle=60.0)
        img_aug.add_random_crop((64,64), 3)
    else:
        print("Do not use data augmentation\n")
        img_aug = tflearn.data_augmentation.ImageAugmentation()

    return img_aug

def image_preprocess():
    img_prep = tflearn.data_preprocessing.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=0.442018)
    img_prep.add_featurewise_stdnorm(std=0.277759)
    return img_prep

def create_net(model, img_prep, img_aug, learning_rate):
    if model == "alex":
        network = alexNet(img_prep, img_aug, learning_rate)
    elif model == "vgg":
        network = vggNet(img_prep, img_aug, learning_rate)
    elif model == "vggtop5":
        network = vggNetTop5(img_prep, img_aug, learning_rate)
    elif model == "vgg13":
        network = vggNet13(img_prep, img_aug, learning_rate)
    elif model == "vgg16":
        network = vggNet16(img_prep, img_aug, learning_rate)
    elif model == "res":
        network = resNet(img_prep, img_aug, learning_rate)
    elif model == "alch11":
        network = alchNet11(img_prep, img_aug, learning_rate)
    elif model == "alch11_without_dropout":
        network = alchNet11(img_prep, img_aug, learning_rate, 0)
    elif model == "alch_enhance":
        network = alchNetEnhance(img_prep, img_aug, learning_rate)
    else:
        network = alchNet19(img_prep, img_aug, learning_rate)
    return network

def main(name, num_epochs, aug_strategy, model):
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    if not os.path.exists('output'):
        os.makedirs('output')
    
    modle = "alch19"

    print("Start: " + name)
    batch_size = 256
    learning_rate = 0.001
    data_dir = "data/tiny-imagenet-200"

    X, Y, X_test, Y_test = get_data(data_dir, model)

    img_prep = image_preprocess()
    img_aug = set_data_augmentation(model, aug_strategy)
    network = create_net(model, img_prep, img_aug, learning_rate)
    checkpoint_path = 'output/'+name+'/'
    sess = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    # sess.fit(X, Y, n_epoch=num_epochs, shuffle=True, validation_set=(X_test, Y_test),
    # show_metric=True, batch_size=batch_size, run_id=name)
    
    network = create_net(model, img_prep, img_aug, learning_rate)
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate,
                         metric=tflearn.metrics.Top_k(k=1)
                        )
    model_1 = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    model_1.load("/home/ruiqixiao77/tflearn-cv/output/alch19-FLIP-72/4778", weights_only=True)
    a = model_1.evaluate(X_test, Y_test, batch_size=batch_size)
    print(a)

    network = create_net(model, img_prep, img_aug, learning_rate)
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate,
                         metric=tflearn.metrics.Top_k(k=5)
                        )
    model_2 = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    model_2.load("/home/ruiqixiao77/tflearn-cv/output/alch19-FLIP-72/4778" + 'checkpoint')
    b = model_2.evaluate(X_test, Y_test, batch_size=batch_size, weights_only=True)
    print(b)
    # import csv
    # with open('results.csv', 'wb') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',')
    #     spamwriter.writerow(name, aug, epoch, model, a, b)

if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='default',
                        help='Name of this training run. Will store results in output/[name]')
    parser.add_argument('--num_epochs', type=int,
                        default=30,
                        help='Name of this training run. Will store results in output/[name]')
    parser.add_argument('--aug_strategy', type=str,
                        default='default',
                        help='Name of this training run. Will store results in output/[name]')
    parser.add_argument('--model', type=str,
                        default='alch',
                        help='model name')
    args, unparsed = parser.parse_known_args()
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    if not os.path.exists('output'):
        os.makedirs('output')
    main(args.name, args.num_epochs, args.aug_strategy, args.model)