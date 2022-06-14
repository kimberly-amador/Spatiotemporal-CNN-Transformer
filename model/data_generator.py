import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


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
            img = np.load(self.imagePath + '{0}_preprocessed.npz'.format(patient[1]))['img']
            lbl = np.load(self.imagePath + '{0}_preprocessed.npz'.format(patient[1]))['label']

            # Extract the desired slice from volume and save data to array
            X[k, ] = img[int(patient[0]), ]
            y[k, ] = lbl[int(patient[0]), ]

        return np.split(X, self.timepoints, axis=3), to_categorical(y, num_classes=self.n_classes)
