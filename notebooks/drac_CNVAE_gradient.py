import tensorflow as tf
from tensorflow import keras
import numpy as np
import generacion_cartoon.visualization.visualize_CNVAE as visualize
import mlflow
import time

def build_encoder(latent_dim, filters=[32, 64], kernel_size=3, activation="relu", input_shape=(56, 56, 1)):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Conv2D(filters=filters[0], kernel_size=kernel_size, strides=2, activation=activation),
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=kernel_size, strides=2, activation=activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim),
    ])

def build_decoder(latent_dim, filters=[128, 64, 1], kernel_size=3, activation="relu", final_activation="sigmoid", reshape_dims=(7,7,100)):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(latent_dim,)),
        tf.keras.layers.Dense(units=reshape_dims[0]*reshape_dims[1]*reshape_dims[2], activation=activation),
        tf.keras.layers.Reshape(target_shape=reshape_dims),
        tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=kernel_size, strides=2, padding="same", activation=activation),
        tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=kernel_size, strides=2, padding="same", activation=activation),
        tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=kernel_size, strides=2, padding="same", activation=final_activation),
    ])

class CNVAE(tf.keras.Model):
    """Convolutional no variational autoencoder."""
    def __init__(self, latent_dim, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self._init_metrics()

    def build(self, input_shape):
        self.input_shape_ = input_shape
        self.encoder = build_encoder(self.latent_dim)
        self.decoder = build_decoder(self.latent_dim)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "beta": self.beta
        })
        return config

    def get_build_config(self):
        return {
            "input_shape": (56, 56, 1)  # o el shape que esperas como entrada
        }

    def build_from_config(self, config):
        self.build(config["input_shape"])

    def _init_metrics(self):
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.center_loss_tracker = keras.metrics.Mean(name="center_loss")
        self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")

    def summary(self):
        print("\nEncoder Summary:")
        self.encoder.summary()

        print("\nDecoder Summary:")
        self.decoder.summary()

        # Total de parámetros
        encoder_params = np.sum([np.prod(v.shape) for v in self.encoder.trainable_variables])
        decoder_params = np.sum([np.prod(v.shape) for v in self.decoder.trainable_variables])
        total_params = encoder_params + decoder_params

        print(f"\nTotal trainable parameters: {total_params:,}")

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
        #grad_imgenc_dec = tape.gradient(tf.reduce_sum(img_decoded), img_encoded)
        #loss += tf.reduce_mean(tf.square(grad))


        grad_decoded_loss = tf.reduce_mean(
            tf.norm(grad_imgenc_dec, axis=-1)
        )  # How much change img_decoded changing img_encoded
            # The magnitud of the grad
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
            reconstruction_loss +
            self.beta*grad_decoded_loss + 
            center_loss
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
    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "input_shape": self.input_shape_,
            # Agrega todos los parámetros que usas en __init__
        })
        return config

    def compile(self, optimizer, patience=100, params=None, path_models=None):
        self.optimizer = optimizer
        self.patience = patience
        self.params = params
        self.path_models = path_models
        self.best_total_loss = float('inf')
        self.count = 0

    def fit(self, train_ds):
        tf.config.run_functions_eagerly(True)

        if self.params["latent_dim"] == 2:
            visualize.plot_latent_images_dim(
                model=self, num_images_x=20, epoch=0, dim=self.params["latent_dim"]
            )

        for epoch in range(1, self.params["epochs"] + 1):
            start_time = time.time()

            loss_total = tf.keras.metrics.Mean()
            loss_recon = tf.keras.metrics.Mean()
            loss_center = tf.keras.metrics.Mean()
            loss_grad = tf.keras.metrics.Mean()

            for idx, train_x in enumerate(train_ds):
                self.train_step(train_x, self.optimizer)
                if epoch == 1 and idx % 75 == 0 and self.params["latent_dim"] == 2:
                    visualize.plot_latent_images_dim(
                        model=self, num_images_x=20, epoch=epoch, first_epoch=True, f_ep_count=idx,
                        dim=self.params["latent_dim"],
                    )
                total_loss_train, recon_loss_train, center_loss_train, grad_loss_train = self.compute_loss(train_x)
                loss_total(total_loss_train)
                loss_recon(recon_loss_train)
                loss_center(center_loss_train)
                loss_grad(grad_loss_train)

            total_loss_train = loss_total.result()
            reconstruction_loss_train = loss_recon.result()
            center_loss_train = loss_center.result()
            grad_decoded_loss_train = loss_grad.result()

            if total_loss_train < self.best_total_loss:
                self.best_total_loss = total_loss_train
                self.save(self.path_models)
                print(f"Best model saved with ELBO: {total_loss_train:.2f} at epoch: {epoch}")
                self.count = 0
            else:
                self.count += 1

            mlflow.log_metric("total_loss_train", total_loss_train, step=epoch)
            mlflow.log_metric("reconstruction_loss_train", reconstruction_loss_train, step=epoch)
            mlflow.log_metric("center_loss_train", center_loss_train, step=epoch)
            mlflow.log_metric("grad_loss_train", grad_decoded_loss_train, step=epoch)

            end_time = time.time()
            print(f"Epoch: {epoch}, total_loss_train: {total_loss_train:.2f}, "
                  f"recons_loss_train: {reconstruction_loss_train:.2f}, "
                  f"grad_loss_train: {grad_decoded_loss_train:.2f}, "
                  f"center_loss_train: {center_loss_train:.2f}, time_epoch: {end_time - start_time:.2f}s")

            if epoch != 1 and self.params["latent_dim"] == 2:
                visualize.plot_latent_images_dim(model=self, num_images_x=20, epoch=epoch, dim=self.params["latent_dim"])

            if self.count >= self.patience:
                print("Early stopping triggered")
                break



