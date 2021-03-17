import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow.keras.layers import *


# building the model with the vectorize layer and the embedding layer
def define_model(EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE):

    # adding an embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        MAX_TOKENS, EMBEDDING_SIZE, input_length=MAX_SEQ_LEN
    )

    # cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    # rnn = tf.keras.layers.RNN(cells)

    # attention_in = tf.keras.layers.LSTM(
    #     100, return_sequences=True, dropout=DROPOUT_RATE
    # )(embedding_layer)

    # attention_out = tf.keras.layers.Attention()(attention_in)

    lstm_1 = tf.keras.layers.LSTM(100)
    dropout = tf.keras.layers.Dropout(DROPOUT_RATE)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([embedding_layer, lstm_1, dropout, output_layer])

    return model


def define_rnn(
    EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT, TYPE_OF_RNN
):
    # defining the input layer
    input_layer = Input(shape=(MAX_SEQ_LEN,))

    # adding an embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        MAX_TOKENS, EMBEDDING_SIZE, input_length=MAX_SEQ_LEN
    )

    # output of the embedding layer
    embedding_layer_out = embedding_layer(input_layer)

    if TYPE_OF_RNN == "LSTM":
        # LSMT layer
        lstm_out = tf.keras.layers.LSTM(
            100,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
            recurrent_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
            bias_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
        )(embedding_layer_out)
        lstm_out = tf.keras.layers.Dropout(DROPOUT_RATE)(lstm_out)
    elif TYPE_OF_RNN == "GRU":
        # LSMT layer
        lstm_out = tf.keras.layers.GRU(
            100,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
            recurrent_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
            bias_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
        )(embedding_layer_out)
        lstm_out = tf.keras.layers.Dropout(DROPOUT_RATE)(lstm_out)

    # attention layer block
    dim = int(lstm_out.shape[2])  # getting shape
    attention_layer = Dense(1, activation="tanh")(lstm_out)
    attention_layer = Flatten()(attention_layer)
    attention_layer = Activation("softmax")(attention_layer)
    attention_layer = RepeatVector(dim)(attention_layer)  # recurring layer
    attention_layer = Permute([2, 1])(attention_layer)
    attention_out = tf.keras.layers.concatenate([lstm_out, attention_layer])
    attention_out = Lambda(
        lambda xin: tf.keras.backend.sum(xin, axis=-2), output_shape=(dim,),
    )(attention_out)
    output = Dense(1, activation="sigmoid")(attention_out)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    return model
