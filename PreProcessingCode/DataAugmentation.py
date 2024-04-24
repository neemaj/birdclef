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
    spectrogram = np.array(spectrogram)
    num_chunks = math.ceil(spectrogram.shape[0] / chunk_size)
    
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
    return np.roll(spectrogram, shift, axis = 1)