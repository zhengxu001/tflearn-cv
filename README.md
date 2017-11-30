# Tutorial code for tensorflow with tflearn

### Installation
1. Follow the installation steps for tensorflow: https://www.tensorflow.org/install/
2. Install tflearn as described at: http://tflearn.org/installation/
3. Install the newest version of h5py if you want to create hdf5 datasets to read from.

### Usage
`python train.py`  
Optional args include:  
* `--data_dir /path/to/tinyimagenet/` to point to the directory where the images are stored. 
By default it expects the data to be inside the *data/* directory, for convenience you could just do a softlink there:  
`cd data`  
`ln -s /path/to/tinyimagenet tiny-imagenet-200`  
* `--hdf5` should be added if you want to create and read from a hdf5 database. Faster file access, but consumes much more space on the harddrive.  
* `--name name_of_training_run`. By default the training run will be named default and will store tensorboard information and trained weights with that
name in the respective *tensorboard/* and *output/* directories. Change that every run if you don't want information to be overwritten.  

### Tensorboard
To launch tensorboard, start it in a terminal with:  
`tensorboard --logdir=tensorboard/`  
By default you can access it now in your browser at `http://127.0.0.1:6006/`  


### Creating a new model
Just copy the *models/baseline_model.py* file and rename it. You can now add or remove layers, change hyperparameters, etc.  
Take a look at the examples and available layers at http://tflearn.org/.  
Also, make sure to import the new model in the *train.py* file.

