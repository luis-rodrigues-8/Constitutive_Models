import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', None)


def plot_the_loss_curve(epochs, mse_training, mse_validation):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(epochs[0:], mse_training[0:], label="Training Loss")
    plt.plot(epochs[0:], mse_validation[0:], label="Validation Loss")
    plt.legend()

    # We're not going to plot the first epoch, since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mse_lists = mse_training[1:] + mse_validation[1:]
    highest_loss = max(merged_mse_lists)
    lowest_loss = min(merged_mse_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()


X = np.load('X_run.npy')
y = np.load('y_run.npy')

# let's save 10% of the data for testing. these curves won't be part of the model training
test_split = 0.1
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
# let's check the array shapes:
print('Input data')
print('Training set: ', x_train.shape)
print('Test set: ', x_test.shape)
print(' ')
print('Features')
print('Training set: ', y_train.shape)
print('Test set: ', y_test.shape)
# train_size = train_df.shape[0]
# test_size = test_df.shape[0]

# reshape for keras training
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 2))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 2))
# shuffle data
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Set the hyperparameters
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.005
BATCH_SIZE = 32
EPOCHS = 40
INPUT_SHAPE = x_train.shape[1:]

# Define the model's architecture
inputs = keras.Input(shape=INPUT_SHAPE)

dense = layers.Dense(16, activation="relu")
x = dense(inputs)

dropout = tf.keras.layers.Dropout(0.2)
x = dropout(x)

x = tf.keras.layers.Flatten()(x)

x = layers.Dense(16, activation="relu")(x)

outputs = layers.Dense(5)(x)


model = keras.Model(inputs=inputs, outputs=outputs, name="model")

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mean_squared_error"],
)

model.summary()


callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]


history = model.fit(
    x_train,
    y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)


epochs = history.epoch
hist = history.history
plot_the_loss_curve(epochs, hist["loss"],
                    hist["val_loss"])

print('\n')
print("Prediction for curve 0: ", model.predict(x_test[0:1]))

print("Real parameters of curve 0: ", y_test[0:1])