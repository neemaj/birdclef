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




