import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp
import glob
import imageio.v2 as imageio
import os
import generacion_cartoon.utils.paths as path
import tensorflow as tf

def plot_original_vs_generated(model, num_img=0, test_ds=None, train_ds=None):
    """
    Plots original and generated images from both train and test datasets.
    
    Args:
        model: The model with `.encoder` and `.decode` methods.
        num_img: Index of the image to display.
        test_ds: Test dataset (batched).
        train_ds: Training dataset (batched).
    """
    # Extract one batch from each dataset
    train_img = next(iter(train_ds))[num_img] if train_ds is not None else None
    test_img = next(iter(test_ds))[num_img] if test_ds is not None else None

    # Generate reconstructions
    gen_train = model.decode(model.encoder(train_img[None, ...])) if train_img is not None else None
    gen_test = model.decode(model.encoder(test_img[None, ...])) if test_img is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    fig.suptitle(f"Original vs Generated", fontsize=14)

    if train_img is not None:
        axes[0, 0].imshow(train_img.numpy().squeeze(), cmap='gray')
        axes[0, 0].set_title("Train - Original")
        axes[0, 1].imshow(gen_train.numpy().squeeze(), cmap='gray')
        axes[0, 1].set_title("Train - Generated")
    else:
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')

    if test_img is not None:
        axes[1, 0].imshow(test_img.numpy().squeeze(), cmap='gray')
        axes[1, 0].set_title("Test - Original")
        axes[1, 1].imshow(gen_test.numpy().squeeze(), cmap='gray')
        axes[1, 1].set_title("Test - Generated")
    else:
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()



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
