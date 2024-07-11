model_path = '/Users/neema/Downloads/final_model.keras'
test_path = '/Users/neema/Downloads/five_birds_test'
pc = '/'

import tensorflow as tf
import numpy as np
import os
import time
import random
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def get_path_label(folder_path, pc):
    '''
        Params:
            file_path: string path to folder containing all bird audio files
            
        Returns:
            dictionary where 
              key = array of audio file paths corresponding to that bird
              value = bird folder name as a string
    '''
    audio_dict = dict()
    bird_folders = os.listdir(Path(folder_path))
    
    ltrain = LabelEncoder()

    ltrain.fit(bird_folders)
    
    

    # Iterate over files in directory
    for bird_path in bird_folders:
        #bird_list is a generator
        bird_list = os.listdir(Path(folder_path + pc + bird_path))
        for f in bird_list:
            #audio_dict[f] = bird_path.name
            
            audio_dict[f] = ltrain.transform([bird_path])[0]
        
    return audio_dict

model = tf.keras.models.load_model(model_path)
labels_dict = get_path_label(test_path, pc)
file_path_list = list()
bird_folders = list()
for entry in os.scandir(test_path):
    if entry.is_dir():
        bird_folders.append(entry)
for bird_path in bird_folders:
    file_path_list.extend(list(Path(bird_path).glob('**/*.npy')))


#making the np array
X = np.empty([len(file_path_list), 512, 256])
y = np.empty([len(file_path_list)])

for i, path in enumerate(file_path_list):
    X[i,] = np.load(path, allow_pickle=True)
    y[i] = labels_dict[path.name]

#evaluating
loss, acc = model.evaluate(X, y, verbose = 2)
print(model.summary())
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print()
print(model.predict(X))
print(model.predict(X).shape)