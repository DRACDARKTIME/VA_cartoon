import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp
import glob
import imageio.v2 as imageio
import os
import generacion_cartoon.utils.paths as path
import tensorflow as tf


def plot_img_original_generated(model, num_img=10, test_ds=None):
    for img in test_ds:
        img_encoded = model.encoder(img)
        generated_image = model.decode(img_encoded)
        plt.imshow(img[num_img], cmap="gray")
        plt.axis("off")
        plt.show()
        plt.imshow(generated_image[num_img].numpy().squeeze(), cmap="gray")
        plt.axis("off")
        plt.show()
        break


def plot_latent_images(
    model,
    num_images_x,
    epoch,
    im_size=56,
    save=True,
    first_epoch=False,
    f_ep_count=0,
    stan_des=1,
    mean=0,
):

    # Create image matrix
    image_width = im_size * num_images_x
    image_height = image_width
    image = np.zeros((image_height, image_width))

    # Create list of values which are evenly spaced wrt probability mass

    norm = tfp.distributions.Normal(mean, stan_des)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, num_images_x))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, num_images_x))

    # For each point on the grid in the latent space, decode and
    # copy the image into the image array
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (im_size, im_size))
            image[i * im_size : (i + 1) * im_size, j * im_size : (j + 1) * im_size] = (
                digit.numpy()
            )

    # Plot the image array
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="Greys_r")
    plt.axis("Off")

    # Potentially save, with different formatting if within first epoch
    if save and first_epoch:
        name = "tf_grid_at_epoch_{:04d}.{:04d}.png".format(epoch, f_ep_count)
        plt.savefig(path.data_created_dir(name))
        plt.close()
    elif save:
        name = "tf_grid_at_epoch_{:04d}.png".format(epoch)
        plt.savefig(path.data_created_dir(name))
        plt.close()


def plot_latent_images_dim(
    model,
    num_images_x,
    epoch,
    im_size=56,
    save=True,
    first_epoch=False,
    f_ep_count=0,
    dim=2,
):
    if dim == 2:
        plot_latent_images(
            model,
            num_images_x,
            epoch,
            im_size=56,
            save=True,
            first_epoch=False,
            f_ep_count=0,
        )
    else:
        pass


def create_gif(
    name_gif,
    path_save_gif=path.data_created_dir(),
    path_images=path.data_created_dir(),
    name_images="tf_grid*.png",
    remove_images=True,
):
    anim_file = os.path.join(path_save_gif, name_gif)
    path_images_name = os.path.join(path_images, name_images)
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = glob.glob(path_images_name)
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    if remove_images == True:
        for file in filenames:
            os.remove(file)


def create_gif_dim(
    name_gif,
    path_save_gif=path.data_created_dir(),
    path_images=path.data_created_dir(),
    name_images="tf_grid*.png",
    remove_images=True,
    dim=2,
):

    if dim == 2:
        create_gif(
            name_gif,
            path_save_gif=path.data_created_dir(),
            path_images=path.data_created_dir(),
            name_images="tf_grid*.png",
            remove_images=True,
        )
    else:
        pass
