from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from BirdGenerator import *
import keras_tuner as kt
import json

#constants
is_neema_mac = True
is_neema = True


def run_small_model(X_train, y_train, X_valid, y_valid):
    '''
    Params:
        X_train: 3D numpy arrays of all final spectrograms (each as its own 2D array) we're training on
        y_train: corresponding labels (idk if they should be strings or ints) 
        
        X_test: 3D numpy arrays of all final spectrograms (each as its own 2D array) we're training on
        y_test: corresponding labels (idk if they should be strings or ints) 
        
    Returns:

    '''
    

    model = Sequential()
    model.add(Conv2D(64, kernel_size = (5,5), strides = (2,1), input_shape = (512, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(Conv2D(64, kernel_size = (5,5), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(Conv2D(128, kernel_size = (5,5), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(Conv2D(256, kernel_size = (5,5), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(Conv2D(256, kernel_size = (3,3), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(X_train.shape[0], activation='softmax'))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Nadam(learning_rate = 0.1), 
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    model.fit(X_train,y_train, epochs = 10)


def run_small_gen_model(X_train, X_valid, label_dict):
    '''
    Params:
        X_train: 3D numpy arrays of all final spectrograms (each as its own 2D array) we're training on
        y_train: corresponding labels (idk if they should be strings or ints) 
        
        label_dict: dictionary with path keys and label values
        
    Returns:

    '''
    # Generators
    training_generator = Bird_Data_Generator(X_train, label_dict, batch_size=8)
    validation_generator = Bird_Data_Generator(X_valid, label_dict, batch_size=3)

    # Design model
    model = Sequential()

    
    model.add(keras.Input(batch_size = training_generator.batch_size, shape=(512, 256, 1)))
    model.add(keras.layers.Dropout(.2))
    
    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(64, kernel_size = (5,5), strides = (2,1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(64, kernel_size = (5,5), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(128, kernel_size = (5,5), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(256, kernel_size = (5,5), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(256, kernel_size = (3,3), strides = (1,1), input_shape = (256, 256, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape = (256, 256, 1)))
    
    
    model.add(Flatten())
    
    model.add(keras.layers.BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(.4))
    model.add(Dense(np.array(X_train).shape[0], activation='softmax'))
    

    # Train model on dataset
    
    
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Nadam(learning_rate = 0.1), 
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    model.fit(training_generator, batch_size =training_generator.batch_size,  validation_data=validation_generator)



def run_small_hp_model(X_train, X_valid, label_dict, path):
    '''
    Params:
        X_train: 3D numpy arrays of all final spectrograms (each as its own 2D array) we're training on
        y_train: corresponding labels (idk if they should be strings or ints) 
        
        label_dict: dictionary with path keys and label values
        
    Returns:

    '''
    training_generator = Bird_Data_Generator(X_train, label_dict, batch_size=3)
    validation_generator = Bird_Data_Generator(X_valid, label_dict, batch_size=3)

    hps = tune_hyperparameters(training_generator, validation_generator, np.array(X_train).shape)

    with open(path, "w") as fp:
        json.dump(hps.values, fp) 
    #keras.models.save_model(model, path)

    

    # Generators
    

'''
def build_model(hp, input_shape, batch_size):
    # Design model
    model = Sequential()
    model.add(keras.Input(batch_size = batch_size, shape=(512, 256, 1)))
    
    
    model.add(layers.Dropout(rate=hp.Float(f'dropout_rate', min_value=0, max_value=0.2, step=0.2)))


    # Add convolutional layers with hyperparameter tuning
    for i in range(hp.Int('num_conv_layers', 3, 5)):  # Number of convolutional layers (3 to 7)

        model.add(layers.Conv2D(
            filters=hp.Choice(f'filters_{i}', values=[32, 64, 128, 256, 512]),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3,4,5]),
            activation='relu'))
        
        model.add(layers.MaxPooling2D(
            pool_size=hp.Choice(f'pool_size_{i}', values=[1, 2,3]),
            strides=hp.Choice(f'stride_{i}', values=[1,2,3])))
        
        
    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=hp.Int('units', min_value=512, max_value=1024, step=256),
        activation='relu'))
    
    model.add(layers.Dropout(rate=hp.Float('dropout_dense', min_value=0, max_value=0.3, step=0.3)))
    model.add(Dense(input_shape[0], activation='softmax'))

    model.compile(
       optimizer =keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),  loss='sparse_categorical_crossentropy')

    return model
    
'''

def build_model(hp, input_shape, batch_size):
    # Design model
    model = Sequential()
    model.add(keras.Input(batch_size = batch_size, shape=(512, 256, 1)))

 # Add convolutional layers with hyperparameter tuning
    for i in range(hp.Int('num_conv_layers', 3, 5)):  # Number of convolutional layers (3 to 7)

        model.add(layers.Conv2D(
            filters=hp.Choice(f'filters_{i}', values=[64, 128, 256]),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3,5]),
            activation='relu'))
        
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,1)))
        
    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=hp.Int('units', min_value=512, max_value=1024, step=256),
        activation='relu'))




    model.add(Dense(input_shape[0], activation='softmax'))

    model.compile(optimizer =keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),  
        loss='sparse_categorical_crossentropy',  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model

    
    
def tune_hyperparameters(train_generator, valid_generator, input_shape):
    if is_neema_mac:
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_shape, batch_size =train_generator.batch_size),
            objective='val_categorical_accuracy',
            max_trials=5,  #Adjust if needed
            executions_per_trial=3,
            directory='/Volumes/Extreme SSD/DS/train_audio_smaller/hyperparam_tuning',
            project_name='bird_classification')
    elif is_neema:
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_shape, batch_size =train_generator.batch_size),
            objective='val_categorical_accuracy',
            max_trials=5,  #Adjust if needed
            executions_per_trial=3,
            directory='D:\\DS\\hyperparam_tuning',
            project_name='bird_classification')
    else:
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_shape, batch_size =train_generator.batch_size),
            objective='val_categorical_accuracy',
            max_trials=5,  #Adjust if needed
            executions_per_trial=3,
            directory='hyperparam_tuning',
            project_name='bird_classification')
    tuner.search(train_generator, epochs=10, validation_data=valid_generator)
    best_hps = tuner.get_best_hyperparameters()[0]

    best_hps.summary()
    tuner.results_summary()

    return best_hps

    







'''
use_multiprocessing=True,
workers=6'''