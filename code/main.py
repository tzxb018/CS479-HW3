import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
import model
import util

# function for running the model
def run_model(
    train_x,
    train_y,
    test_x,
    test_y,
    TYPE_OF_RNN,
    EMBEDDING_SIZE,
    MAX_TOKENS,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EPOCHS,
    DROPOUT_RATE,
    REG_CONSTANT,
):
    # train learning algorithm L1 on training set i to get hypo 1
    # getting our model
    model1 = model.define_rnn(
        EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT, TYPE_OF_RNN
    )

    # compiling our model!
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    # setting up early stopping
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_accuracy", mode="max", min_delta=1, baseline=0.4,
    # )

    # training our model
    history = model1.fit(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        # callbacks=[early_stopping],
        validation_split=0.1,
    )

    util.print_arch(model1)

    # saving our model
    PATH = (
        "rnn_"
        + str(TYPE_OF_RNN)
        + "_dr"
        + str(DROPOUT_RATE).replace(".", "x")
        + "_rc"
        + str(REG_CONSTANT).replace(".", "x")
    )
    # model.save(PATH)
    tf.keras.Model.save(model1, "./models/" + PATH)

    # evaluating our model
    evaluate = model1.evaluate(test_x, test_y, verbose=0)
    print("test accuracy: ", evaluate[1])
    print("loss: ", evaluate[0])

    return evaluate[0], evaluate[1], history


# ***********************************MAIN*****************************************
# # getting the data set
MAX_SEQ_LEN = 128
MAX_TOKENS = 5000
(x_train, y_train), (x_test, y_test) = util.getDataset(MAX_TOKENS, MAX_SEQ_LEN)

# hyperparameters
# hyperparams = [
#     [[0, 0.01], [0.5, 0.01]],
#     [[0, 0.01], [0, 0.1]],
#     [[0.5, 0.01], [0.5, 0.1]],
#     [[0.5, 0.01], [0, 0.1]],
#     [[0, 0.1], [0.5, 0.1]],
#     [[0, 0.01], [0.5, 0.1]],
# ]

# for params in hyperparams:
#     DROPOUT_RATE_1 = params[1][0]
#     REG_CONSTANT_1 = params[1][1]
#     DROPOUT_RATE_2 = params[0][0]
#     REG_CONSTANT_2 = params[0][1]
DROPOUT_RATE_1 = 0
REG_CONSTANT_1 = 0.1
DROPOUT_RATE_2 = 0
REG_CONSTANT_2 = 0.1

# setting up k-fold cross validation
K = 30
model1_accuracies = []
model2_accuracies = []
model1_loss = []
model2_loss = []
error_diff = []
error_diff_estimation = 0

for i in range(1, K):

    print("K-iteration " + str(i) + "*********************************")
    len_of_train = len(x_train)
    len_of_test = len(x_test)
    # partitioning data into K equal sized subsets
    # since dataset is already divided into training and testing, we will partition both the training and testing data seperately
    x_train_k = x_train[(len_of_train // K * (i - 1)) : (len_of_train // K * i)]
    y_train_k = y_train[(len_of_train // K * (i - 1)) : (len_of_train // K * i)]
    x_test_k = x_test[(len_of_test // K * (i - 1)) : (len_of_test // K * i)]
    y_test_k = y_test[(len_of_test // K * (i - 1)) : (len_of_test // K * i)]

    print(len(x_train_k), len(x_test_k))

    BATCH_SIZE = 64
    EPOCHS = 50
    EMBEDDING_SIZE = 32
    TYPE_OF_RNN_1 = "LSTM"

    # running model 1
    eval_loss, eval_acc, history = run_model(
        x_train_k,
        y_train_k,
        x_test_k,
        y_test_k,
        TYPE_OF_RNN_1,
        EMBEDDING_SIZE,
        MAX_TOKENS,
        MAX_SEQ_LEN,
        BATCH_SIZE,
        EPOCHS,
        DROPOUT_RATE_1,
        REG_CONSTANT_1,
    )

    model1_accuracies.append(eval_loss)
    model1_loss.append(eval_acc)

    # *********************************MODEL 2******************************************

    TYPE_OF_RNN_2 = "GRU"

    # train learning algorithm L2 on training set i to get hypo 2
    eval_loss2, eval_acc2, history2 = run_model(
        x_train_k,
        y_train_k,
        x_test_k,
        y_test_k,
        TYPE_OF_RNN_2,
        EMBEDDING_SIZE,
        MAX_TOKENS,
        MAX_SEQ_LEN,
        BATCH_SIZE,
        EPOCHS,
        DROPOUT_RATE_2,
        REG_CONSTANT_2,
    )

    model2_accuracies.append(eval_acc2)
    model2_loss.append(eval_loss2)

    # finding the error difference in this k-iteration
    p_i = eval_acc - eval_acc2
    error_diff.append(p_i)
    error_diff_estimation = error_diff_estimation + p_i

error_diff_estimation = error_diff_estimation / K
print("Error Difference Estaimation: " + str(error_diff_estimation))
f = open("./output/evaluated_results.txt", "a")
f.write(
    "\nModel "
    + str(TYPE_OF_RNN_1)
    + "1: Dropout rate: "
    + str(DROPOUT_RATE_1)
    + " Reg Constant: "
    + str(REG_CONSTANT_1)
    + " Model "
    + str(TYPE_OF_RNN_2)
    + "2: Dropout rate: "
    + str(DROPOUT_RATE_2)
    + " Reg Constant: "
    + str(REG_CONSTANT_2)
)
f.write("\nEstimated Diff: " + str(error_diff_estimation) + "\n")
f.close()

# evaluating both models on whole dataset
EPOCHS = 3
eval_full_loss1, eval_full_acc1, history_full_1 = run_model(
    x_train,
    y_train,
    x_test,
    y_test,
    TYPE_OF_RNN_1,
    EMBEDDING_SIZE,
    MAX_TOKENS,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EPOCHS,
    DROPOUT_RATE_1,
    REG_CONSTANT_1,
)
print("finished with model 1\n")

eval_full_loss2, eval_full_acc2, history_full_2 = run_model(
    x_train,
    y_train,
    x_test,
    y_test,
    TYPE_OF_RNN_2,
    EMBEDDING_SIZE,
    MAX_TOKENS,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EPOCHS,
    DROPOUT_RATE_2,
    REG_CONSTANT_2,
)
print("finished with model 2\n")

# graphing the two full trained model
# util.graph_two(
#     history_full_1,
#     history_full_2,
#     DROPOUT_RATE_1,
#     DROPOUT_RATE_2,
#     REG_CONSTANT_1,
#     REG_CONSTANT_2,
# )

# print("finished graphing")

# getting the confusion matrix from both models

PATH1 = (
    "rnn_"
    + str(TYPE_OF_RNN_1)
    + "_dr"
    + str(DROPOUT_RATE_1).replace(".", "x")
    + "_rc"
    + str(REG_CONSTANT_1).replace(".", "x")
)

PATH2 = (
    "rnn_"
    + str(TYPE_OF_RNN_2)
    + "_dr"
    + str(DROPOUT_RATE_2).replace(".", "x")
    + "_rc"
    + str(REG_CONSTANT_2).replace(".", "x")
)

util.get_confusion_matrix(PATH1, x_test, y_test, DROPOUT_RATE_1, REG_CONSTANT_1)
print("confusion matrix of model 1")

util.get_confusion_matrix(PATH2, x_test, y_test, DROPOUT_RATE_2, REG_CONSTANT_2)
print("confusion matrix of model 2")

