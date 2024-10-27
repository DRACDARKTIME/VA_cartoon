import tensorflow as tf
import keras
import numpy as np


class CNVAE(tf.keras.Model):
    """Convolutional no variational autoencoder."""

    def __init__(self, latent_dim, alpha=1.0, beta=1.0, gamma=1.0):
        super(CNVAE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.center_loss_tracker = keras.metrics.Mean(name="center_loss")
        self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(56, 56, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=16,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="sigmoid",
                ),
            ]
        )

    def decode(self, z):
        """Image decoded without activation"""
        img_decoded = self.decoder(z)
        return img_decoded

    @tf.function
    def sample(self, z=None):
        """Take a sample from our latent space"""
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z)

    def reparameterize(self, img_lat_spa):
        """Random vector of latent space
        z = alpha + epsilon*latent_space"""
        eps = tf.random.normal(shape=mean.shape)
        alpha = tf.random.normal(shape=mean.shape)
        return eps * img_lat_spa + alpha

    def compute_loss(self, x):
        """The loss is a combination of two terms:
        the reconstruction term (how well the reconstructed image resembles the original image)
        and
        the regularization term (how well the latent space follows a normal distribution)
        """
        with tf.GradientTape() as tape:
            img_encoded = self.encoder(x)
            # z = self.reparameterize(img_lat_spa)
            img_decoded = self.decode(img_encoded)

        grad_imgenc_dec = tape.gradient(img_decoded, img_encoded)

        grad_decoded_loss = tf.reduce_mean(
            tf.norm(grad_imgenc_dec, axis=-1)
        )  # How much change img_decoded changing img_encoded
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x, img_decoded), axis=(1, 2)
            )
        )  # how much images look alike
        center_loss = tf.reduce_mean(tf.reduce_sum(tf.square(img_encoded), axis=-1))
        total_loss = (
            self.alpha * reconstruction_loss
            + self.beta * grad_decoded_loss
            + self.gamma * center_loss
        )
        return total_loss, reconstruction_loss, center_loss, grad_decoded_loss

    @tf.function
    def train_step(self, x, optimizer):
        """Executes one training step and returns the loss.
        This function computes the loss and gradients, and uses the latter to
        update the ,model's parameters.
        """
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, center_loss, grad_decoded_loss = (
                self.compute_loss(x)
            )
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.center_loss_tracker.update_state(center_loss)
        self.grad_loss_tracker.update_state(grad_decoded_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "center_loss": self.center_loss_tracker.result(),
            "grad_loss": self.grad_loss_tracker.result(),
        }
