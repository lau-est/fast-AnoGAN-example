import os
import glob
import random
import numpy as np
from metrics import *

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

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


# Funtion to convert from tensor to image
def tensor2image(tensor):
    image = 0.5*( tensor.data.cpu().float().numpy() + 1 )
    return image


# Function to visualize the training losses of WGAN using tensorboard
def show_tensorboard(input_img, output_img, loss, epoch, tensorboard):
    input_img  = tensorimage(input_img)
    output_img = tensorimage(output_img)
    diff_img   = output_img - input_img
    diff_img   = (diff_img)*(diff_img)

    tensorboard.add_scalar('Training/Generator',      loss[0], epoch + 1)
    tensorboard.add_scalar('Training/Discriminator',  loss[1], epoch + 1)
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


# Save score of anomalies
def save_scores_anom(img_distance, anomaly_score, z_distance, dir_to_save):
    df = pd.DataFrame(img_distance)
    df.to_csv(dir_to_save + '/img_distance.csv')

    df = pd.DataFrame(anomaly_score)
    df.to_csv(dir_to_save + '/anomaly_score.csv')

    df = pd.DataFrame(z_distance)
    df.to_csv(dir_to_save + '/z_distance.csv')


# Save latent space of the real images
def save_csv_z_real(data_z, dir_to_save):
    np.save(dir_to_save + '/latent_space_Z_real', data_z)


# Save ñatent space of the generated images
def save_csv_z_fake(data_z, dir_to_save):
    np.save(dir_to_save + '/latent_space_Z_fake', data_z)


# Function to save results
def save_results(list_input, list_output, img_size, dir_to_save):
    
    create_checkdirectory(dir_to_save)

    #print(list_input)
    list_mse  = []
    list_psnr = []
    list_ssim = []
    list_wass = []
    list_frec = []


    for i in range(len(list_input)):
        input_img  = tensor2image(list_input[i])
        output_img = tensor2image(list_output[i])
        diff_img   = output_img - input_img
        diff_img   = (diff_img) * (diff_img)

        
        original  = np.reshape(input_img,  (img_size,img_size))
        output    = np.reshape(output_img, (img_size,img_size))
        
        # MSE
        mse_loss  = mean_squared_error(original, output)

        # PSNR
        psnr = calculate_psnr(original, output) 

        # SSIM
        ssim_loss = ssim(original, output, data_range=output.max() - output.min())

        # WASSERSTEIN
        wass_loss = wasserstein_distance(original.flatten(), output.flatten())
        
        # FRECHET
        frec_loss = calculate_fid(original, output)


        list_mse.append(mse_loss)
        list_psnr.append(psnr)
        list_ssim.append(ssim_loss)
        list_wass.append(wass_loss)
        list_frec.append(frec_loss)
        
        input_img  = np.reshape(input_img,  (img_size,img_size,1))
        output_img = np.reshape(output_img, (img_size,img_size,1))
        diff_img   = np.reshape(diff_img,   (img_size,img_size,1))

        
        cv.imwrite(dir_to_save + '/o-%d.png' % (i+1),  input_img  * 255)
        cv.imwrite(dir_to_save + '/r-%d.png' % (i+1),  output_img * 255)
        cv.imwrite(dir_to_save + '/s-%d.png' % (i+1),  diff_img   * 255)

    
    df = pd.DataFrame(list_mse)
    df.to_csv(dir_to_save + '/mse.csv')

    df1 = pd.DataFrame(list_psnr)
    df1.to_csv(dir_to_save + '/psnr.csv')

    df2 = pd.DataFrame(list_ssim)
    df2.to_csv(dir_to_save + '/ssim.csv')

    df3 = pd.DataFrame(list_wass)
    df3.to_csv(dir_to_save + '/wass.csv')

    df4 = pd.DataFrame(list_frec)
    df4.to_csv(dir_to_save + '/frec.csv')
