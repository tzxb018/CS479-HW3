import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential


def build_discriminator(in_shape=(32, 32, 3)):

    # modelled after discriminator defined in Brownlee's work
    discriminator = Sequential()
    discriminator.add(Conv2D(32, (3, 3), padding="same", input_shape=in_shape))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(1, activation="sigmoid"))

    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=0.0002, beta_1=0.5),
        metrics=["accuracy"],
    )
    discriminator.summary()
    return discriminator


def build_generator(latent_dim):

    # modelled off infoGAN
    generator = Sequential()

    generator.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))
    generator.add(Reshape((8, 8, 128)))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())
    generator.add(Conv2D(128, kernel_size=3, padding="same"))
    generator.add(Activation("relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())
    generator.add(Conv2D(64, kernel_size=3, padding="same"))
    generator.add(Activation("relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv2D(3, kernel_size=3, padding="same"))
    generator.add(Activation("tanh"))
    generator.summary()
    return generator


def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    gan.compile(loss="binary_crossentropy", optimizer=Adam())
    return gan
