import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, Input, Concatenate, GlobalMaxPooling1D, Dropout
from Transformer import TransformerEncoder, PositionalEmbedding


class ENCODER:
    @staticmethod
    def build(reg=l2(0.00005), shape=(384, 256), init='he_normal'):
        # Create the model
        i = Input(shape=(*shape, 1))

        # The first two layers will learn a total of 16 filters with a 3x3 kernel size
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(i)
        d1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d1')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d1)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 32 total learned filters
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d2 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d2')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d2)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 64 total learned filters
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d3')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d3)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 128 total learned filters
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d4 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d4')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d4)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 256 total learned filters
        o = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d5 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d5')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d5)  # Downsampling

        # Encoder outputs
        size = o.shape[1] * o.shape[2]
        output = Reshape((1, size))(o)
        down = [d1, d2, d3, d4, d5]
        return Model(inputs=i, outputs=[output, *down], name='Encoder')


class TRANSFORMER:
    @staticmethod
    def build(inputTensor, timepoints, key_dim=96, n_heads=8, n_layers=1):
        x = PositionalEmbedding(timepoints, inputTensor.shape[2], name='frame_position_embedding')(inputTensor)
        for _ in range(n_layers):
            x = TransformerEncoder(inputTensor.shape[2], key_dim, n_heads)(x)
        x = GlobalMaxPooling1D()(x)
        output = Dropout(0.5)(x)
        return output


class DECODER:
    @staticmethod
    def build(inputTensor, down, n_classes, reg=l2(0.00005), init='he_normal'):
        # Reshape to fit the decoder
        # TODO: automate reshape dimensions according to encoder's output
        o = Reshape((12, 8, 1))(inputTensor)

        u1 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u1, down[4]])
        # The first layers will learn a total of 256 filters with a 3x3 kernel size
        o = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u2 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u2, down[3]])
        # Stack two more layers, keeping the size of each filter as 3x3 but decreasing to 128 total learned filters
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u3 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u3, down[2]])
        # Stack two more layers, keeping the size of each filter as 3x3 but decreasing to 64 total learned filters
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u4 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u4, down[1]])
        # Stack two more layers, keeping the size of each filter as 3x3 but decreasing to 32 total learned filters
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u5 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u5, down[0]])
        # Stack three more layers, two 16 filter layers and a softmax layer
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        output = Conv2D(n_classes, 1, padding="same", activation='softmax', name='Softmax')(o)
        return output


# -----------------------------------
#      EXTRA: DATA GENERATOR
# -----------------------------------

class DataGenerator2D(Sequence):
    """
    To utilize this data generator:
        1. Images should be already preprocessed, saved as a numpy array of name 'PatientID/preprocessed.npz'
        2. Npz files must contain two variables named 'img' and 'label', where 'img' is the CTP data and 'label' is the
        ground truth for the corresponding patient
        3. 'img' should be of shape (n_slices, height, width, timepoints)
        4. 'label' should be of shape (n_slices, height, width, 1)
        5. Elements in the patient list (list_IDs_temp) should have the format: '#_PatientID', where # is an
        integer referring to the slice number

    Output:
        - List of length=timepoints. Each element on the list contains an array of shape (batch_size, height, width, 1)
        - Single array containing the one-hot-encoded label -- array shape: (batch_size, height, width, n_classes)
    """
    def __init__(self, list_IDs, imagePath, dim=(384, 256), batch_size=1, timepoints=32, n_classes=2, shuffle=True, **kwargs):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.timepoints = timepoints
        self.n_classes = n_classes
        self.imagePath = imagePath
        self.shuffle = shuffle
        self.on_epoch_end()

    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes]  # Find list of IDs
        X, y = self.__data_generation(list_IDs_temp)  # Generate data
        return X, y

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.timepoints), dtype='float32')
        y = np.empty((self.batch_size, *self.dim, 1), dtype='float32')

        # Generate data according to patient IDs
        for k, ID in enumerate(list_IDs_temp):
            patient = ID.split('_', maxsplit=1)
            img = np.load(self.imagePath + '{0}/preprocessed.npz'.format(patient[1]))['img']
            lbl = np.load(self.imagePath + '{0}/preprocessed.npz'.format(patient[1]))['label']

            # Extract the desired slice from volume and save data to array
            X[k, ] = img[int(patient[0]), ]
            y[k, ] = lbl[int(patient[0]), ]

        return np.split(X, self.timepoints, axis=3), to_categorical(y, num_classes=self.n_classes)
