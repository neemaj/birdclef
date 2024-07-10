import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import binary_erosion, binary_dilation,generate_binary_structure


def mask(spectrogram):
    '''Params:
            spectrogram:2D numpy array of spectrogram
        Returns:
            returns tuple of numpy spectrograms masks as numpy arrays (signal,inverse noise)
    '''
    
    signal_spec= np.zeros(spectrogram.shape)
    noise_spec= np.zeros(spectrogram.shape)
    
    row_meds = np.zeros(spectrogram.shape[0])
    col_meds = np.zeros(spectrogram.shape[1])
    
    for row_i in range(spectrogram.shape[0]):
        current_row = spectrogram[row_i,:]
        row_meds[row_i] = np.median(current_row)
        
    for col_i in range(spectrogram.shape[1]):
        current_col = spectrogram[:,col_i]
        col_meds[col_i] = np.median(current_col)
        
    for row_i in range(spectrogram.shape[0]):
        row_median =row_meds[row_i]
        for col_i in range(spectrogram.shape[1]):
            col_median = col_meds[col_i]
            current_pixel = spectrogram[row_i,col_i]
            
            if current_pixel > col_median*3 and current_pixel > row_median*3:
                signal_spec[row_i,col_i] = 1
                
            if current_pixel > col_median*2.5 and current_pixel > row_median*2.5:
                noise_spec[row_i,col_i] = 1
                
    return signal_spec, noise_spec


def apply_binary_erosion_and_dilation_spectrogram(spectrogram):
    '''
    Params:
        spectrogram: 2D numpy array of spectrogram
    Return:
        2D numpy array with binary values corresponding to binary eroded and dilated spectrogram
    '''
    struct1 = generate_binary_structure(2, 16)
    eroded = binary_erosion(spectrogram, structure = struct1, border_value = 0)
    dilated = binary_dilation(eroded, structure = struct1)
    return dilated.astype(np.double)


def get_slices_indicator(spectrogram):
    '''
    Params:
        spectrogram: 2D numpy array of spectrogram
    Return:
        1D numpy array (len = seconds) (1 = nonzero column for that second)
    '''
    num_seconds = spectrogram.shape[0]
    indicator = np.zeros(num_seconds)
    
    for row_i in range(num_seconds):
        current_row = spectrogram[row_i]
        if(np.max(current_row) == 1):
            indicator[row_i] = 1
    return indicator
        

def apply_binary_dilation_indicator(indicator):
    '''
    Params:
        indicator: 1D numpy array of indicator
    Return:
        1D numpy array of indicator after 2 dilations
    '''
    #rank= 1 (num dims)
    #connectivity = 4 (4x1 kernel)
    struct1 = generate_binary_structure(1, 4)
    dilation = binary_dilation(indicator, structure = struct1, iterations = 2)
    
    return dilation

def plot_indicator(indicator): 
    '''
    Params:
        indicator: 1D numpy array of indicator
    
    '''
    f, ax = plt.subplots(1, figsize=(8, 4))
    plt.scatter(np.arange (1, indicator.shape[0] + 1), indicator, s=5)
    plt.title('Signal Indicator')
    plt.xlabel('Time')
    plt.ylabel('Indicator')
    ax.set_xlim(xmin=0, xmax = indicator.shape[0])
    plt.show()
    
