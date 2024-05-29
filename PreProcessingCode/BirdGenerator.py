# help on how to create a bird generator received from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# much of the code is an altered version of their generator

#WHAT WE NEED: list of file paths, dictionary of file paths and their corresponding label
import numpy as np
from tensorflow import keras

class Bird_Data_Generator(keras.utils.Sequence):

    def __init__(self, file_paths, labels, freq_bins=256, chunk_length=512, batch_size=8, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.freq_bins = freq_bins
        self.chunk_length = chunk_length
        self.on_epoch_end()

    def __len__(self):
        len_value = int(np.floor(len(self.file_paths) / self.batch_size))
        print(len_value)
        return len_value

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        file_paths_temp = [self.file_paths[k] for k in indexes]
        
        return self.__data_generation(file_paths_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_paths_temp):
        #X = np.empty((self.batch_size, self.chunk_length,self.freq_bins))
        X = np.empty([self.batch_size, 512, 256])
        y = np.empty([self.batch_size])

        for i, path in enumerate(file_paths_temp):
            X[i,] = np.load(path, allow_pickle=True)
            y[i] = self.labels[path.name]

        print(f'shape is {X.shape}')
        return X, y
        
        
    