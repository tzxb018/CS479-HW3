import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
import model
import util
import random

# code adapted from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
def train(generator, discriminator, gan, dataset, latent_dim):
    EPOCHS = 200
    BATCH = 128
    epoch_batch_size = int(dataset.shape[0] / BATCH)
    batch_half_size = int(BATCH / 2)

    for i in range(EPOCHS):
        for j in range(epoch_batch_size):
            # generate samples (both fake and real)
            (x_real, y_real), (x_fake, y_fake) = util.generate_samples(
                dataset, generator, latent_dim, batch_half_size
            )

            loss_real, acc_real = discriminator.train_on_batch(x_real, y_real)
            loss_fake, acc_fake = discriminator.train_on_batch(x_fake, y_fake)

            gan_input = util.generate_latent_points(latent_dim, BATCH)
            gan_output = np.ones((BATCH, 1))

            gan_loss = gan.train_on_batch(gan_input, gan_output)

        print(
            "[INFO] EPOCH %d: loss_real=%.3f, loss_fake=%.3f, gan_loss=%.3f"
            % (i + 1, loss_real, loss_fake, gan_loss)
        )
        if (i + 1) % 10 == 0:
            util.print_eval(generator, discriminator, dataset, latent_dim, i)


# MAIN ( https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/)

# size of the latent space
LATENT_DIM = 100
# create the discriminator
discriminator = model.define_discriminator()
# create the generator
generator = model.define_generator(LATENT_DIM)
# create the gan
gan_model = model.define_gan(generator, discriminator)
# load image data
dataset_celeb = util.load_dataset()
# train model
train(generator, discriminator, gan_model, dataset_celeb, LATENT_DIM)
