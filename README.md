# Tutorial code for tensorflow with tflearn

## Data
You can download the tiny-imagenet data from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip).  
The dataset contains 100000 images for training inside the *train/* folder and 10000 images for validation inside the *val/* folder.

## Prerequisites
* Ubuntu (14 or higher)
* Cuda 8
* cuDNN (optional, will speed up training.)
* python-pip
* h5py (optional, only if you want to use the `--hdf5` option, see below.)
* virtualenv (optional)

## Installation

### Step 1: Install CUDA and cuDNN
Download CUDA 8 and cuDNN library from the following links. The default website takes one to CUDA 9. The previous versions can be found in "Legacy Releases" under "Additional Resources".
* [Cuda 8](https://developer.nvidia.com/cuda-80-ga2-download-archive)
* [cuDNN 7 for cuda 8](https://drive.google.com/a/tcd.ie/file/d/1JNKUnIRbAnZ49wSiJgou4zz9CQiamR8J/view?usp=sharing)

For CUDA, follow the installation instructions from the website. For cuDNN, they ask you to create an account in order to download the library. You don't have to since we have downloaded it for you. You can just follow the installation intructions mentioned [here](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

### Step 2 : Creating a virtual environment (optional)
[Why use virtual environment ?](https://pythontips.com/2013/07/30/what-is-virtualenv/)

Run the following commands
```
virtualenv -p /path/to/python2.7 project_name
source project_name/bin/activate
```
"/path/to/python2.7" is usually /usr/bin/python2.7. 
Once you activate a virtual environment, you will be able to use only the local packages. Check this link if you wish to know how to use the global packages. 
[Make virtualenv inherit specific packages from your global site-packages](https://stackoverflow.com/questions/12079607/make-virtualenv-inherit-specific-packages-from-your-global-site-packages)

### Step 3: Install tensorflow and tflearn
1. Follow the installation steps for [tensorflow](https://www.tensorflow.org/install/).
2. Install tflearn as described [here](http://tflearn.org/installation/).
3. Install the newest version of h5py if you want to create hdf5 datasets to read from: `sudo pip install h5py`

## Usage
To train the baseline_model, just run the *train.py* script:  
```python train.py```  
Optional arguments include:  
* `--data_dir /path/to/tinyimagenet/` to point to the directory where the images are stored. 
By default it expects the data to be inside the *data/* directory, for convenience you could just do a softlink there:  
`cd data`  
`ln -s /path/to/tinyimagenet tiny-imagenet-200`  
* `--hdf5` should be added if you want to create and read from a hdf5 database. Faster file access, but consumes much more space on the harddrive.  
* `--name name_of_training_run`. By default the training run will be named default and will store tensorboard information and trained weights with that
name in the respective *tensorboard/* and *output/* directories. Change that every run if you don't want information to be overwritten.  

## Tensorboard
To launch tensorboard, start it in a terminal with:  
`tensorboard --logdir=tensorboard/`  
By default you can access it now in your browser at `http://127.0.0.1:6006/`  

## Creating a new model
Just copy the *models/baseline_model.py* file and rename it. You can now add or remove layers, change hyperparameters, etc.  
Take a look at the examples and available layers [here](http://tflearn.org/). You can also change some training options in *train.py*, for example add more data augmentation options. 
Also, make sure to import the new model in the *train.py* file.

