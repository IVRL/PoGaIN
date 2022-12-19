"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.
"""

import os
import sys
from abc import abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import utils


def fix_gpu():  # Needed to add this to make TF work on my GPU
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    return InteractiveSession(config=config)


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        filenames,
        seed,
        batch_size,
        shuffle,
    ):
        "Initialization"
        self.filenames = filenames
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.filenames))
        self.rng = np.random.default_rng(seed)
        self.bsds_path_train = os.path.join(
            os.path.dirname(__file__), "../../../BSDS300/images/train"
        )

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        batch_filenames = [self.filenames[k] for k in indexes]

        # Generate data
        inputs, ab_values = self.data_generation(batch_filenames)

        return inputs, ab_values

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    @abstractmethod
    def data_generation(self, batch_filenames):
        pass


class Generator(DataGenerator):
    def __init__(
        self,
        filenames,
        seed,
        batch_size,
        dim,
        n_channels=1,
        shuffle=True,
    ):
        super().__init__(
            filenames,
            seed,
            batch_size,
            shuffle,
        )

        self.dim = dim
        self.n_channels = n_channels

    def data_generation(self, batch_filenames):
        "Generates data containing batch_size samples"
        # Initialization
        inputs = np.empty((self.batch_size, *self.dim, self.n_channels))
        ab_values = np.empty((self.batch_size, 2))

        for i, filename in enumerate(batch_filenames):
            s = self.rng.integers(low=0, high=10)
            a = self.rng.uniform(low=1, high=100)
            b = self.rng.uniform(low=0.01, high=0.15)

            f = os.path.join(self.bsds_path_train, filename)
            gt, shape = utils.load_image(f)

            x = utils.add_noise(gt, a, b, seed=s)
            x = np.reshape(x, shape)[0 : self.dim[0], 0 : self.dim[0]]

            inputs[i] = x[:, :, np.newaxis]
            ab_values[i] = [a, b]

        inputs = tf.stack([tf.convert_to_tensor(i) for i in inputs], axis=0)
        ab_values = tf.convert_to_tensor(ab_values)

        return inputs, ab_values


def model(inputShape, filters):
    chanDim = -1
    inputs = Input(shape=inputShape)
    x = inputs

    # Loop over the number of filters
    for f in filters:
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    x = Dense(2, activation="linear")(x)

    return Model(inputs, x)


def train_cnn(
    seed, dim, epochs, batch_size, validation_percentage, model_type, filters, loss_type
):
    session = fix_gpu()

    bsds_path_train = os.path.join(
        os.path.dirname(__file__), "../../../BSDS300/images/train"
    )
    filenames = os.listdir(bsds_path_train)

    print("Prepare training...")
    filenames_train, filenames_val = train_test_split(
        filenames, test_size=validation_percentage, random_state=seed
    )

    gen_train = Generator(filenames_train, seed, batch_size, dim)
    gen_val = Generator(filenames_val, seed, batch_size, dim)

    model = model_type(inputShape=(*dim, 1), filters=filters)
    opt = Adam()
    model.compile(loss=loss_type, optimizer=opt)

    print("Training model...")
    model.fit(
        x=gen_train,
        validation_data=gen_val,
        epochs=epochs,
    )

    model.save(os.path.join(os.path.dirname(__file__), "model"))

    session.close()
