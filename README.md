# Fast AnoGAN Method

Simple f-AnoGAN method for unsupervised anomaly detection on images.

## Requirements


### Create a virtual environment

Install virtualenv and virtualenvwrapper.

        $ sudo pip install virtualenv virtualenvwrapper
        $ sudo rm -rf ~/get-pip.py ~/.cache/pip

To finish the install we need to update our ~/.bashrc file.

        $ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
        $ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
        $ echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
        $ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc

Then, source the ~/.bashrc file.

        $ source ~/.bashrc

Creating a virtual environment to hold the requirements. THis line simply creates a Python3 virtual environment named mypy.

        $ mkvirtualenv mypy -p python3

Let's activate the mypy environment by using the workon command.

        $ workon mypy


### Install requirements from setup.py

Use the setup.py file to install all the requirements into the mypy virtual environment.

        $ python setup.py install


## Data Structure
The dataset directory must be located into the main folder **data**. It contains the training and testing dataset images, and it is organized as follow:

        data
            ├──   training/
            │     ├── img_001.png
            |     ├── img_002.png        
            │     ├── img_003.png
            |     ├── img_004.png 
            |     ├── ***_***.png
            |
            ├──   testing/
            |     ├── img_001.png        
            │     ├── img_002.png
            |     ├── img_003.png
            |     ├── img_004.png
            |     ├── ***_***.png
            └── 

Please, note that images should have a 256 x 256 dimension.

## F-AnoGAN Training

The f-AnoGAN method needs to train two components: the WGAN and the encoder.
Therefore, you must train the WGAN firstly, and the the encoder part.

### Important training settings to perform the experiments.

* Experiment name        : --exp_name
* Epochs number          : --n_epochs
* Latent space dimension : --latent_dim
* Validation percentage  : --valid_split

### Training the WGAN component

        $ python train_wgan.py  --exp_name exp1 --n_epochs 500 --latent_dim 100 --batch_size 4 --data_root data/


### Training the encoder component

        $ python train_encoder.py  --exp_name exp1 --n_epochs 500 --latent_dim 100 --batch_size 4 --data_root data/


Once executing the WGAN or Encoder components training, it creates two directories automatically: **fit_model** and **runs**. The first one stores the fitted models, and the second one stores the files for tensorboard visualization. 

Therefore you can trade of the status of the training by visualizing the tensorboard:

        $ cd runs
        $ tensorboard --logdir runs



## F-AnoGAN Testing

### Important testing settings to perform the experiments

* Experiment name        : --exp_name
* Latent space dimension : --latent_dim

### Testing

        $ python test.py --exp_name exp1 --latent_dim 100 --data_root data/


Once executed test.py, it creates the **results directory** automatically. It stores the results of the generated images and the csv files with the evaluation metrics.

**NOTE: --exp_name and --latent_dim values must be the same in the training and testing stages to avoid issues.** 
