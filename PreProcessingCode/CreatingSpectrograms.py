#code source: https://www.tensorflow.org/tutorials/audio/simple_audio#convert_waveforms_to_spectrograms

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def get_spectrogram(audio_tensor):
    '''Returns a spectogram
    Params: 
      audio_tensor: 1D tensor that contains audio
    Return: 2D tensor where rows== frequencies, cols = time, values = absolute value amplitude
    '''
    #make each frame 25 ms
    frame_size = 0.025
    
    #75% overlap between frames
    frame_stride = 0.00625 
    
    #32Hz sampling rate for bird audio files
    sample_rate = 32000
    
    #samples per window length
    frame_length = frame_size * sample_rate
    
    #samples per window step
    frame_step=frame_stride * sample_rate
    
    #Frequency Resolution = max frequency/fft_length
    #max frequency = sample_rate/2
    #16000/1024 = 15.625 Hz <-- size of frequency band we differentiate between
    fft_length = 1024
  
    # Convert the audio_tensor to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
    audio_tensor, frame_length=int(frame_length), frame_step=int(frame_step), fft_length = fft_length)
    
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    
    
    return spectrogram

#OLD AND DEPRECATED
#plots the log frequencies on y-axis and time on x axis
def plot_spectrogram(spectrogram, ax):
    '''
        plots the spectrogram with the log of the frequency vs time
        Params:
            spectrogram: 2D tensor of spectrogram
            ax: axes to be plotted on
    '''
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time in seconds')
    ax.set_ylabel('log(Freq in Hz)')

def plot_log_spectrogram(spectrogram, title):
    '''
    Params:
        spectrogram: numpy array of spectrogram
        title: name of spectrogram
    '''
    plt.figure(figsize=(10, 4))
    plt.imshow(tf.math.log(spectrogram.T + 1e-6), aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    
def plot_abs_spectrogram(spectrogram, title):
    '''
    Params:
        spectrogram: numpy array of spectrogram
        title: name of spectrogram
    '''
    f, ax = plt.subplots(1, figsize=(10, 4))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()


def min_max_scale_spectrogram(spectrogram):
    '''
    Params:
        spectrogram: 2D tensor representing the spectrogram where each row is a collection
                    of amplitudes at various frequencies at a certain time
    Returns: 2D spectrogram tensor with all amplitudes scaled from 0 to 1 using the max ampliitude
                for that spectrogram
    '''

    largestmax = 0
    for row in spectrogram:
        currentmax = tf.reduce_max(row)
        if (currentmax > largestmax):
            largestmax = currentmax


    return tf.divide(spectrogram, largestmax)


def log_scale_spectrogram(spectrogram):
    '''
    Params:
        spectrogram: 2D tensor representing the spectrogram where each row is a collection
                    of amplitudes at various frequencies at a certain time
    Returns: 2D spectrogram tensor with all amplitudes log scaled
    '''
    log_spectrogram = tf.convert_to_tensor(np.log((spectrogram.numpy()).T + np.finfo(float).eps))

    
    
    return log_spectrogram




