import os
import glob
import random
import numpy as np

# Function to create a directory
def create_checkdirectory(checkpoint_directory):
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)


# Function to split the dataset into training and validation
def split_path(dir, valid_split, mode = "train"):
    files       = glob.glob(os.path.join(dir, '%s' % mode) + '/*.*')
    
    random.shuffle(files)

    n_files = len(files)
    n_split = int(n_files * valid_split)

    
    return files[n_split:], files[:n_split]


# Function to convert from tensor to image
def tensorimage(tensor):
    image = 0.5*( tensor.data.cpu().float().numpy() + 1 )
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image


# Function to visualize the training losses of WGAN using tensorboard
def show_tensorboard(input_img, output_img, loss, epoch, tensorboard):
    input_img  = tensorimage(input_img)
    output_img = tensorimage(output_img)
    diff_img   = output_img - input_img
    diff_img   = (diff_img)*(diff_img)

    tensorboard.add_scalar('Training/Generator',     loss[0], epoch + 1)
    tensorboard.add_scalar('Training/Discriminator', loss[1], epoch + 1)
    tensorboard.add_scalar('Training/Reconstruction', loss[2], epoch + 1)
    tensorboard.add_image ('Training_Results/Input',               input_img)
    tensorboard.add_image ('Training_Results/Output',             output_img)
    tensorboard.add_image ('Training_Results/Diff'  ,               diff_img)


# Function to visualize the validation losses of WGAN using tensorboard
def show_tensorboard_valid(input_img, output_img, loss, epoch, tensorboard):
    input_img  = tensorimage(input_img)
    output_img = tensorimage(output_img)
    diff_img   = output_img - input_img
    diff_img   = (diff_img)*(diff_img)

    tensorboard.add_scalar('Validation/Generator',     loss[0], epoch + 1)
    tensorboard.add_scalar('Validation/Discriminator', loss[1], epoch + 1)
    tensorboard.add_scalar('Validation/Reconstruction', loss[2], epoch + 1)
    tensorboard.add_image ('Validation_Results/Input',               input_img)
    tensorboard.add_image ('Validation_Results/Output',             output_img)
    tensorboard.add_image ('Validation_Results/Diff'  ,               diff_img)


# Function to visualize the training loss of the Encoder using tensorboard
def show_tensorboard_enc(input_img, output_img, loss, epoch, tensorboard):
    input_img  = tensorimage(input_img)
    output_img = tensorimage(output_img)
    diff_img   = output_img - input_img
    diff_img   = (diff_img)*(diff_img)

    tensorboard.add_scalar('Training/Encoder',     loss[0], epoch + 1)
    tensorboard.add_image ('Training_Results/Input',               input_img)
    tensorboard.add_image ('Training_Results/Output',             output_img)
    tensorboard.add_image ('Training_Results/Diff'  ,               diff_img)


# Function to visualize the validation loss of the Encoder using tensorboard
def show_tensorboard_enc_val(input_img, output_img, loss, epoch, tensorboard):
    input_img  = tensorimage(input_img)
    output_img = tensorimage(output_img)
    diff_img   = output_img - input_img
    diff_img   = (diff_img)*(diff_img)

    tensorboard.add_scalar('Validation/Encoder',     loss[0], epoch + 1)
    tensorboard.add_image ('Validation_Results/Input',               input_img)
    tensorboard.add_image ('Validation_Results/Output',             output_img)
    tensorboard.add_image ('Validation_Results/Diff'  ,               diff_img)

