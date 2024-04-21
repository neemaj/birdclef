Resources
https://www.altexsoft.com/blog/audio-analysis/


ML Tools for Audio
- Tensorflow-io package: 
used for preprocessing (noise removal, waveforms → spectrogram, frequency, time-masking)
- Easy to start ML model training with tensorflow
https://www.tensorflow.org/io/tutorials/audio



Pre-Processing
- Framing: cutting a continuous stream into shorter pisces (frames)
- Windowing: fixes a spectral leakage issue caused during Fourier transform. Usually done using Hanning window function
- Windowing: fixes a spectral leakage issue caused during Fourier transform. Usually done using Hanning window function https://medium.com/@milana.shxanukova15/n-fft-parameter-in-fft-134712e1c79d
- Overlap-add (OLA): preserves information caused by windowing

Feature Extraction
- Time-domain
- Amplitude vs. Time 
- Waveform graphs
- Amplitude envelope: tracks amplitude peaks. Used in onset detection (tracking when a new note starts)
- short-time energy (STE): used for identifying energy (high amplitude and frequency) variation. useful for identifying silence
- Zero-crossing Rate (ZCR): Measures how many times the signal wave crosses horizontal axis in frame. .used for identifying silence

Frequency domain
- mean/median frequency
- signal to noise ratio (SNR): compares frequency to background noise
- band energy ratio (BER): measures how low frequencies are dominant over high ones.
- Time-Frequency Domain Features
- spectrogram obtained by using short-time fourier transformation
- mel-frequency cepstral coefficients (MFCCs): used in analyzing human speech
- Time-frequency: process of applying weights to the bins of a time-frequency representation to enhance, diminish, or isolate portions of audio.


- NFFT
  > frequency resolution = sample_rate / n_fft
  > frequency resolution: how small a gap between freq are we differentiating
  > sample_rate = 32,000 Hz
  > NFFT of size N yields N/2 Frequencies
  > https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12223
- Window
  > small window focuses on frequencies in a small window of time





ML Models
- by creating spectrograms, we can treat the problem like image recognition
long short-term memory networks (LSTM): known for their ability to spot long-term dependencies in data and remember information from numerous prior steps
- Convolutional Neural Networks (CNN)
- Audio Spectogram Transformer (AST): https://github.com/YuanGongND/ast?tab=readme-ov-file#Introduction


TO DO:

- research OTHER BIRD SONG IDENTIFICATION PROJECTS (https://www.kaggle.com/competitions/birdsong-recognition/discussion/183208)
- research more about time-frequency domain features
- research more ml models



