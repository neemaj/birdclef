import numpy as np
import math
def chunk_and_pad(spectrogram, chunk_size):
    '''
    method used to split the spectrogram into chunks and pad the end
    
        Params:
            spectrogram: numpy array to be split up
            chunk_size: the length of each chunk

        Return:
            list of chunked up numpy arrays
    '''
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = np.array(spectrogram)
        
    num_chunks = math.ceil(spectrogram.shape[0] / chunk_size)
    if num_chunks == 0:
        num_chunks = 1
    
    if spectrogram.shape[0] % chunk_size != 0: #we need to pad
        num_zeros = num_chunks*chunk_size - spectrogram.shape[0]
        #print(num_zeros)
        spectrogram = np.pad(spectrogram, ((0,num_zeros), (0, 0)), mode='constant', constant_values=0)
        
    return np.split(spectrogram, num_chunks)

def time_shift(spectrogram, shift):
    '''
    Method used to shift over spectrogram on time axis
    
        Params:
            spectrogram: numpy array to shift
            shift: amount to time shift by (do random number for a random time shift)
            
        Return:
            time shifted numpy array
    '''
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = np.array(spectrogram)
    return np.roll(spectrogram, shift, axis = 0)

def pitch_shift(spectrogram, shift):
    '''
    Method used to shift over spectrogram on freq axis
    
        Params:
            spectrogram: numpy array to shift
            shift: amount to freq shift by (do random number for a random freq shift)
            
        Return:
            freq shifted numpy array
    '''
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = np.array(spectrogram)
    return np.roll(spectrogram, shift, axis = 1)


def reduce_amplitude(spectrogram, factor):
    '''
    Method used to reduce the amplitude of a spectrogram.

    Params:
        spectrogram: numpy array of spectrogram
        factor: scalar by which to reduce the amplitude (0 < factor < 1)

    Return:
        Amplitude reduced numpy array
    '''
    return spectrogram * factor

'''
#Testing Code
reduced_spectrogram = reduce_amplitude(spectrogram, reduction_factor)

print(reduced_spectrogram)  
'''