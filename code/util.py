import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
import os, time, random
import matplotlib as plt
from numpy.random import randn


def load_dataset():

    # determines if there is a prexisting npy file to read from
    if os.path.exists("./np_celeb_3200.npy"):
        print("existing np file found...")
        np_train = np.load("./np_celeb_3200.npy")
        print("loaded in existing npy...")
        return np_train

    # loads in dataset
    print("loading new dataset...")
    data = tfds.load("celeb_a")
    train_ds = data["train"]

    # converts batch images into a numpy array and appends to train_images
    train_images = []
    i = 0
    max_len = 3200
    for batch in train_ds:
        train_image = batch["image"].numpy()

        # preprocessing to resize image and normalize
        train_image = tf.image.resize(train_image, [32, 32])
        train_image = (train_image - 127.5) / 127.5

        train_images.append(train_image)

        # setting a limit for how many are being read in
        i = i + 1
        print(i)
        if i >= max_len:
            break

    # converting into np array
    print("converting to np array...")
    np_train = np.array(train_images)
    np.save("np_celeb_3200", np_train)

    return np_train


# # code adapted from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/


# genereating the samples needed for the GAN to train
def generate_samples(dataset, generator_model, latent_dim, NUM_SAMPLES):

    # generating the real samples from the dataset
    real_images = dataset[np.random.randint(0, dataset.shape[0], NUM_SAMPLES)]
    real_labels = np.ones((NUM_SAMPLES, 1))

    # generating the fake samples from the generator model
    generator_input = np.random.randn(latent_dim * NUM_SAMPLES)
    generator_input = generator_input.reshape(NUM_SAMPLES, latent_dim)
    fake_images = generator_model.predict(generator_input)
    fake_labels = np.zeros((NUM_SAMPLES, 1))

    # returns a pair of image and labels, one for real and one for fake
    return (real_images, real_labels), (fake_images, fake_labels)


# taken directly from source
# create and save a plot of generated images
def generate_images(image_ex, epoch, size=5):
    # scale from [-1,1] to [0,1]
    image_ex = (image_ex + 1) / 2.0
    # plot images
    for i in range(size * size):
        # define subplot
        plt.pyplot.subplot(size, size, 1 + i)
        # turn off axis
        plt.pyplot.axis("off")
        # plot raw pixel data
        plt.pyplot.imshow(image_ex[i])
    # save plot to file
    filename = "./output/image_of_faces_epoch%03d.png" % (epoch + 1)
    plt.pyplot.savefig(filename)
    plt.pyplot.close()


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, sample_size):

    x_input = np.random.rand(latent_dim * sample_size)
    x_input = x_input.reshape(sample_size, latent_dim)
    return x_input


def print_eval(generator, discriminator, dataset, latent_dim, epoch):

    # retrieving the samples for the discriminator to evaluate on
    NUM_SAMPLES = 150
    (real_images, real_labels), (fake_images, fake_labels) = generate_samples(
        dataset, generator, latent_dim, NUM_SAMPLES
    )

    # evaluating the real and fake samples with the discriminator
    fake_loss, fake_acc = discriminator.evaluate(fake_images, fake_labels, verbose=0)
    real_loss, real_acc = discriminator.evaluate(real_images, real_labels, verbose=0)

    file_name = "./models/generator_model_%03d" % (epoch + 1)
    generator.save(file_name)
    generate_images(fake_images, epoch)

    print(
        "[EVAL] Accuracy of real images: %.0f%%, Accuracy of fake images: %.0f%%"
        % (real_acc * 100, fake_acc * 100)
    )

