import tensorflow as tf
import keras
import numpy as np
import generacion_cartoon.visualization.visualize_CNVAE as visualize
import mlflow
import time


class CustomUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size):
        super(CustomUpSampling2D, self).__init__()
        if type(size) is not tuple and type(size) is not list:
            size = (size, size)
        self.size = size

    def build(self, input_shape):
        pass

    def call(self, input):
        return tf.repeat(tf.repeat(input, self.size[0], axis=1), self.size[1], axis=2)


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
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
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
                # tf.keras.layers.Conv2DTranspose(
                #    filters=16,
                #    kernel_size=3,
                #    strides=1,
                #    padding="same",
                #    activation="relu",
                # ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="sigmoid",
                ),
                # CustomUpSampling2D(size=(2, 2)),
                # tf.keras.layers.Conv2D(
                #    filters=64, kernel_size=3, padding="same", activation="relu"
                # ),
                # CustomUpSampling2D(size=(2, 2)),
                # tf.keras.layers.Conv2D(
                #    filters=32, kernel_size=3, padding="same", activation="relu"
                # ),
                # CustomUpSampling2D(size=(2, 2)),
                # tf.keras.layers.Conv2D(
                #    filters=16, kernel_size=3, padding="same", activation="relu"
                # ),
                # tf.keras.layers.Conv2D(
                #    filters=1, kernel_size=3, padding="same", activation="sigmoid"
                # ),
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
            # img_encoded = self.reparameterize(img_encoded)
            img_decoded = self.decode(img_encoded)

        grad_imgenc_dec = tape.gradient(img_decoded, img_encoded)

        grad_decoded_loss = tf.reduce_mean(
            tf.norm(grad_imgenc_dec, axis=-1)
        )  # How much change img_decoded changing img_encoded
        # reconstruction_loss = tf.reduce_mean(
        #    tf.reduce_sum(tf.keras.losses.MSE(x, img_decoded), axis=(1, 2))
        # )
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
            "total_loss_train": self.total_loss_tracker.result(),
            "recons_loss_train": self.reconstruction_loss_tracker.result(),
            "center_loss_train": self.center_loss_tracker.result(),
            "grad_loss_train": self.grad_loss_tracker.result(),
        }

    def compile(
        self,
        patience=100,
        train_ds=None,
        optimizer=None,
        best_total_loss=100000,
        count=0,
        test_ds=None,
        params=None,
        path_models=None,
    ):
        tf.config.run_functions_eagerly(True)
        visualize.plot_latent_images_dim(
            model=self, num_images_x=20, epoch=0, dim=params["latent_dim"]
        )
        for epoch in range(1, params["epochs"] + 1):
            start_time = time.time()
            loss4 = tf.keras.metrics.Mean()
            loss5 = tf.keras.metrics.Mean()
            loss6 = tf.keras.metrics.Mean()
            loss7 = tf.keras.metrics.Mean()
            for idx, train_x in enumerate(train_ds):
                self.train_step(train_x, optimizer)

                if epoch == 1 and idx % 75 == 0:
                    visualize.plot_latent_images_dim(
                        model=self,
                        num_images_x=20,
                        epoch=epoch,
                        first_epoch=True,
                        f_ep_count=idx,
                        dim=params["latent_dim"],
                    )
                (
                    total_loss_train,
                    reconstruction_loss_train,
                    center_loss_train,
                    grad_decoded_loss_train,
                ) = self.compute_loss(train_x)
                loss4(total_loss_train)
                loss5(reconstruction_loss_train)
                loss6(center_loss_train)
                loss7(grad_decoded_loss_train)

            end_time = time.time()
            total_loss_train = loss4.result()
            reconstruction_loss_train = loss5.result()
            center_loss_train = loss6.result()
            grad_decoded_loss_train = loss7.result()

            loss = tf.keras.metrics.Mean()
            loss1 = tf.keras.metrics.Mean()
            loss2 = tf.keras.metrics.Mean()
            loss3 = tf.keras.metrics.Mean()

            for test_x in test_ds:
                total_loss, reconstruction_loss, center_loss, grad_decoded_loss = (
                    self.compute_loss(test_x)
                )
                loss(total_loss)
                loss1(reconstruction_loss)
                loss2(center_loss)
                loss3(grad_decoded_loss)
            total_loss = loss.result()
            reconstruction_loss = loss1.result()
            center_loss = loss2.result()
            grad_decoded_loss = loss3.result()
            if total_loss < best_total_loss:
                best_total_loss = total_loss
                self.save_weights(path_models)
                print(
                    "Best model saved with best ELBO: {:.2f} in epoch: {}".format(
                        total_loss, epoch
                    )
                )
                count = 0
            else:
                count = count + 1
            mlflow.log_metric("total_loss_test", total_loss, step=epoch)
            mlflow.log_metric("reconstruction_loss", reconstruction_loss, step=epoch)
            mlflow.log_metric("center_loss", center_loss, step=epoch)
            mlflow.log_metric("grad_decodad_loss", grad_decoded_loss, step=epoch)

            mlflow.log_metric("total_loss_train", total_loss_train, step=epoch)
            mlflow.log_metric(
                "reconstruction_loss_train", reconstruction_loss_train, step=epoch
            )
            mlflow.log_metric("center_loss_train", center_loss_train, step=epoch)
            mlflow.log_metric(
                "grad_decodad_loss_train", grad_decoded_loss_train, step=epoch
            )
            print(
                "Epoch: {}, total_loss_train: {:.2f}, recons_loss_train: {:.2f}, grad_loss_train: {:.2f}, center_loss_train: {:.2f}".format(
                    epoch,
                    total_loss_train,
                    reconstruction_loss_train,
                    grad_decoded_loss_train,
                    center_loss_train,
                )
            )
            print(
                "Epoch: {}, Test total_loss: {:.2f}, recons_loss: {:.2f}, grad_loss: {:.2f}, center_loss: {:.2f}, time_epoch: {:.2f}".format(
                    epoch,
                    total_loss,
                    reconstruction_loss,
                    grad_decoded_loss,
                    center_loss,
                    end_time - start_time,
                )
            )
            if epoch != 1:
                visualize.plot_latent_images_dim(
                    model=self, num_images_x=20, epoch=epoch, dim=params["latent_dim"]
                )
            if count == patience:
                break


class FCNVAE(tf.keras.Model):
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
                    filters=128,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
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
                # tf.keras.layers.Conv2DTranspose(
                #    filters=16,
                #    kernel_size=3,
                #    strides=2,
                #    padding="same",
                #    activation="relu",
                # ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="sigmoid",
                ),
                # CustomUpSampling2D(size=(2, 2)),
                # tf.keras.layers.Conv2D(
                #    filters=64, kernel_size=3, padding="same", activation="relu"
                # ),
                # CustomUpSampling2D(size=(2, 2)),
                # tf.keras.layers.Conv2D(
                #    filters=32, kernel_size=3, padding="same", activation="relu"
                # ),
                # CustomUpSampling2D(size=(2, 2)),
                # tf.keras.layers.Conv2D(
                #    filters=16, kernel_size=3, padding="same", activation="relu"
                # ),
                # tf.keras.layers.Conv2D(
                #    filters=1, kernel_size=3, padding="same", activation="sigmoid"
                # ),
            ]
        )
