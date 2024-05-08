from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

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




