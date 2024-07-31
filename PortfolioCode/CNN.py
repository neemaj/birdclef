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





def run_final_model_1(X_train, X_valid, label_dict, model_save_path, number_of_classes, batch_size, chunk_shape =(512, 256, 1)):
    '''
    Builds and runs our CNN and saves the model
    
    Params:
        X_train: list of paths to npy files corresponding to training spectrograms
        X_valid: list of paths to npy files corresponding to validation spectrograms
        
        label_dict: Dictionary mapping file paths to one-hot sparse encoded label values

        model_save_path: where to save the model to

        number_of_classes: number of classes (species of birds)

        batch_size: batch size for generators
        
    

    '''
    #Generators are used to give a batch of npy arrays for stochastic gradient descent function
    training_generator = Bird_Data_Generator(X_train, label_dict, batch_size)
    validation_generator = Bird_Data_Generator(X_valid, label_dict, batch_size)

    # Buildling CNN based on hyperparamter tuning
    model = Sequential()

    model.add(keras.Input(batch_size, chunk_shape))

    model.add(Conv2D(128, kernel_size = (5,5), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,1)))

    model.add(Conv2D(128, kernel_size = (5,5), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,1)))

    model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,1)))

    model.add(Conv2D(256, kernel_size = (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,1)))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))

    model.add(Dense(number_of_classes, activation='softmax'))

    # Train model on dataset
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer =keras.optimizers.Adam(learning_rate=.001),  
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    model.fit(training_generator, batch_size =training_generator.batch_size,  validation_data=validation_generator, epochs = 10)

    model.save(model_save_path)





def run_small_hp_model(X_train, X_valid, label_dict, model_output_path, number_of_classes, batch_size):
    '''
    calls hyperparameter tuning functions and outputs results in json file
    
    Params:
        X_train: list of paths to npy files corresponding to training spectrograms
        X_valid: list of paths to npy files corresponding to validation spectrograms

        label_dict: Dictionary mapping file paths to one-hot sparse encoded label values

        model_save_path: where to save the model to

        number_of_classes: number of classes (species of birds)

        batch_size: batch size for generators
        
    Returns:

    '''
    training_generator = Bird_Data_Generator(X_train, label_dict, batch_size)
    validation_generator = Bird_Data_Generator(X_valid, label_dict, batch_size)

    hps = tune_hyperparameters(training_generator, validation_generator, np.array(X_train).shape,number_of_classes)

    with open(model_output_path, "w") as fp:
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

def build_model(hp, input_shape, batch_size, number_of_classes):
    # Design model
    model = Sequential()
    model.add(keras.Input(batch_size = batch_size, shape=(512, 256, 1)))

 # Add convolutional layers with hyperparameter tuning
    for i in range(3):  # Number of convolutional layers (3 to 7)

        model.add(layers.Conv2D(
            filters=hp.Choice(f'filters_{i}', values=[64, 128]),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3,5]),
            activation='relu'))
        
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,1)))

    model.add(layers.Conv2D(
        filters=hp.Choice(f'filters_3', values=[64, 128, 256]),
        kernel_size=hp.Choice(f'kernel_size_3', values=[3,5]),
        activation='relu'))
        
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,1)))
        
    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=hp.Int('units', min_value=128, max_value=512, step=128),
        activation='relu'))




    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(optimizer =keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),  
        loss='sparse_categorical_crossentropy',  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model



    
    
def tune_hyperparameters(train_generator, valid_generator, input_shape, number_of_classes):
    if is_neema_mac:
        stop_early = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5)
        tuner = kt.Hyperband(
            lambda hp: build_model(hp, input_shape, batch_size =train_generator.batch_size, number_of_classes=number_of_classes),
            objective='val_sparse_categorical_accuracy',
            #max_trials=5,  #Adjust if needed
            #executions_per_trial=1,
            directory='/Volumes/home/SanDisk/DS/hyperparam_tuning',
            project_name='validation_bird_classification')
    elif is_neema:
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_shape, batch_size =train_generator.batch_size,number_of_classes=number_of_classes),
            objective='val_categorical_accuracy',
            max_trials=5,  #Adjust if needed
            executions_per_trial=3,
            directory='D:\\DS\\hyperparam_tuning',
            project_name='bird_classification')
    else:
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_shape, batch_size =train_generator.batch_size,number_of_classes=number_of_classes),
            objective='val_categorical_accuracy',
            max_trials=5,  #Adjust if needed
            executions_per_trial=3,
            directory='hyperparam_tuning',
            project_name='bird_classification')
    tuner.search(train_generator, epochs=1, validation_data=valid_generator, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters()[0]

    tuner.results_summary()

    return best_hps
