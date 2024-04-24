#code source:https://github.com/tensorflow/tensorflow/pull/27140

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp


spectrogram.numpy()

#Calculates median value of spectogram tensor
median_value = tfp.stats.percentile(spectrogram, 50.0)

#Creates binary mask
binary_mask = tf.cast(spectrogram > median_value, tf.float32)

#Applies mask to spectogram
masked_spectrogram = tf.multiply(spectrogram, binary_mask)


def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(tf.math.log(spectrogram + 1e-6).numpy(), aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# Plot the original spectrogram
plot_spectrogram(spectrogram, 'Original Spectrogram')

# Plot the masked spectrogram
plot_spectrogram(masked_spectrogram, 'Masked Spectrogram')
