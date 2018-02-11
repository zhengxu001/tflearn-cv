"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script trains a simple baseline_model on the tiny-imagenet dataset."""

import sys
import os
import argparse
import tflearn
import h5py

sys.path.insert(0, os.path.dirname(__file__))

from datasets.tiny_imagenet import *
from models.baseline_model import *

from tflearn.data_utils import shuffle

def get_data(data_dir, hdf5):
    train_file, val_file = build_dataset_index(data_dir)
    if hdf5:
        # Create folder to store dataset.
        if not os.path.exists('hdf5'):
            os.makedirs('hdf5')
        # Check if hdf5 databases already exist and create them if not.
        if not os.path.exists('hdf5/tiny-imagenet_train.h5'):
            from tflearn.data_utils import build_hdf5_image_dataset
            print('Creating hdf5 train dataset.')
            build_hdf5_image_dataset(train_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_train.h5', categorical_labels=True, normalize=True)

        if not os.path.exists('hdf5/tiny-imagenet_val.h5'):
            from tflearn.data_utils import build_hdf5_image_dataset
            print(' Creating hdf5 val dataset.')
            build_hdf5_image_dataset(val_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_val.h5', categorical_labels=True, normalize=True)

        # Load training data from hdf5 dataset.
        h5f = h5py.File('hdf5/tiny-imagenet_train.h5', 'r')
        X = h5f['X']
        Y = h5f['Y']

        # Load validation data.
        h5f = h5py.File('hdf5/tiny-imagenet_val.h5', 'r')
        X_test = h5f['X']
        Y_test = h5f['Y']    

    # Load images directly from disk when they are required.
    else:
        from tflearn.data_utils import image_preloader
        X, Y = image_preloader(train_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)
        X_test, Y_test = image_preloader(val_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)

    # Randomly shuffle the dataset.
    X, Y = shuffle(X, Y)

    return X, Y, X_test, Y_test


def main(data_dir, hdf5, name):
    """This is the main function of the file.

    Args:
        data_dir: The root directory of the images.
        hdf5: Boolean if a hdf5 database should be created to load in the images.
        name: Name of the current training run."""

    # Set some variables for training.
    batch_size = 256
    num_epochs = 60
    learning_rate = 0.001

    # Load in data.
    X, Y, X_test, Y_test = get_data(data_dir, hdf5)

    # Define some preprocessing options. These will be done on every image before either training or testing.
    img_prep = tflearn.data_preprocessing.ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Define some data augmentation options. These will only be done for training.
    img_aug = tflearn.data_augmentation.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    # img_aug.add_random_blur (sigma_max=5.0)
    # img_aug.add_random_90degrees_rotation(rotations=[0, 1, 2, 3])
    img_aug.add_random_rotation (max_angle=60.0)
    img_aug.add_random_crop((64, 64), 3)

    # Get the network definition.
    network = create_network(img_prep, img_aug, learning_rate)

    # Training. It will always save the best performing model on the validation data, even if it overfits.
    checkpoint_path = 'output/'+name+'/'
    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    model.fit(X, Y, n_epoch=num_epochs, shuffle=True, validation_set=(X_test, Y_test),
    show_metric=True, batch_size=batch_size, run_id=name)
    import csv
    from PIL import Image, ImageOps
    import numpy as np
    with open('eggs.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow("imid,n01443537,n01629819,n01641577,n01644900,n01698640,n01742172,n01768244,n01770393,n01774384,n01774750,n01784675,n01855672,n01882714,n01910747,n01917289,n01944390,n01945685,n01950731,n01983481,n01984695,n02002724,n02056570,n02058221,n02074367,n02085620,n02094433,n02099601,n02099712,n02106662,n02113799,n02123045,n02123394,n02124075,n02125311,n02129165,n02132136,n02165456,n02190166,n02206856,n02226429,n02231487,n02233338,n02236044,n02268443,n02279972,n02281406,n02321529,n02364673,n02395406,n02403003,n02410509,n02415577,n02423022,n02437312,n02480495,n02481823,n02486410,n02504458,n02509815,n02666196,n02669723,n02699494,n02730930,n02769748,n02788148,n02791270,n02793495,n02795169,n02802426,n02808440,n02814533,n02814860,n02815834,n02823428,n02837789,n02841315,n02843684,n02883205,n02892201,n02906734,n02909870,n02917067,n02927161,n02948072,n02950826,n02963159,n02977058,n02988304,n02999410,n03014705,n03026506,n03042490,n03085013,n03089624,n03100240,n03126707,n03160309,n03179701,n03201208,n03250847,n03255030,n03355925,n03388043,n03393912,n03400231,n03404251,n03424325,n03444034,n03447447,n03544143,n03584254,n03599486,n03617480,n03637318,n03649909,n03662601,n03670208,n03706229,n03733131,n03763968,n03770439,n03796401,n03804744,n03814639,n03837869,n03838899,n03854065,n03891332,n03902125,n03930313,n03937543,n03970156,n03976657,n03977966,n03980874,n03983396,n03992509,n04008634,n04023962,n04067472,n04070727,n04074963,n04099969,n04118538,n04133789,n04146614,n04149813,n04179913,n04251144,n04254777,n04259630,n04265275,n04275548,n04285008,n04311004,n04328186,n04356056,n04366367,n04371430,n04376876,n04398044,n04399382,n04417672,n04456115,n04465501,n04486054,n04487081,n04501370,n04507155,n04532106,n04532670,n04540053,n04560804,n04562935,n04596742,n04597913,n06596364,n07579787,n07583066,n07614500,n07615774,n07695742,n07711569,n07715103,n07720875,n07734744,n07747607,n07749582,n07753592,n07768694,n07871810,n07873807,n07875152,n07920052,n09193705,n09246464,n09256479,n09332890,n09428293,n12267677".split(","))
        for i in range(10000):
            img = Image.open("/home/zen/tflearn-demo/test/val/test_data_raw/" + str(i) + ".jpg").convert('RGB')
            img = ImageOps.fit(img, ((64,64)), Image.ANTIALIAS)
            img_arr = np.array(img)
            img_arr = img_arr.reshape(1,64,64,3).astype("float")
            pred = model.predict(img_arr)
            a = np.insert(pred[0], 0, i)
            spamwriter.writerow(a)

if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/tiny-imagenet-200',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--hdf5',
                        help='Set if hdf5 database should be created.',
                        action='store_true')
    parser.add_argument('--name', type=str,
                        default='default',
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    if not os.path.exists('output'):
        os.makedirs('output')
    main(args.data_dir, args.hdf5, args.name)