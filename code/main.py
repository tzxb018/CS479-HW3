import numpy as np  
import tensorflow as tf  
import tensorflow_datasets as tfds  
import matplotlib.pyplot as plt 
from tqdm import tqdm  
import model
import util
import random

# code adapted from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
def train_model(gan, generator, discriminator, dataset, latent_dim):
    EPOCHS = 200
    BATCH = 128
    epoch_batch_size = int(dataset.shape[0] / BATCH)
    batch_half_size = int(BATCH / 2)

    for epoch in range(0, EPOCHS):
        for epoch_batch in range(0, epoch_batch_size):
            # generate samples (both fake and real)
            (x_real, y_real), (x_fake, y_fake) = util.get_fake_and_real_images_and_labels(
                dataset, generator, latent_dim, batch_half_size
            )

            loss_real, acc_real = discriminator.train_on_batch(x_real, y_real)
            loss_fake, acc_fake = discriminator.train_on_batch(x_fake, y_fake)

            gan_input = util.generate_latent_points(latent_dim, BATCH)
            gan_output = np.ones((BATCH, 1))

            gan_loss = gan.train_on_batch(gan_input, gan_output)

        print(
            "[INFO] EPOCH %d: loss_real=%.4f, loss_fake=%.4f, gan_loss=%.4f"
            % (epoch + 1, loss_real, loss_fake, gan_loss)
        )
        if (epoch + 1) % 10 == 0:
            util.print_eval(generator, discriminator, dataset, latent_dim, epoch)


# MAIN
LATENT_DIM = 100
discriminator = model.build_discriminator()
generator = model.build_generator(LATENT_DIM)
gan_model = model.build_gan(generator, discriminator)
dataset = util.load_dataset()
train_model(gan_model, generator, discriminator, dataset, LATENT_DIM)
