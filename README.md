# Under-studied species in the Western Ghats audio identification 🐦

## Objective 
In this project, we created a convolutional neural network (CNN) to identify bird species in the Western Ghats, a biodiversity hotspot.
Our approach included:
- Implement preprocessing steps to handle to handle recording variables, signal noise separation, and chunk division.
- Achieve high accuracy.
- Employ signal/noise separation and chunk division to generate input features for the neural networks.


## Table Of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Data Augmentation](#data-augmentation)
- [Results](#results)
- [Improvements](#improvments)

## Dataset
Our data was used from the BirdClef Kaggle competition. We had training data that had short recordings of bird calls.
More info can be found here:
Kaggle competition: [https://www.kaggle.com/competitions/birdclef-2024/overview](https://www.kaggle.com/competitions/birdclef-2024/overview)



## Files for each step
- [Preprocessing](https://github.com/NeemaJ/BirdCLEF/tree/24f35caf402121c3238eb0dae2cf49cb92c70a81/PreProcessingCode)
- [CNN](https://github.com/NeemaJ/BirdCLEF/blob/f42915c6f4ab40295a891dcf2a1a576fe4a6e0e0/PreProcessingCode/CNN.py)
- [Spectograms](https://github.com/NeemaJ/BirdCLEF/blob/a1e5dbf8a7c2b12c5545801b47eb784638a90152/PreProcessingCode/CreatingSpectrograms.py)
- [Data Augmentation](https://github.com/NeemaJ/BirdCLEF/blob/fe6eeb7da4fc249a3e0625111098da92f996805e/PreProcessingCode/DataAugmentation.py)
- [Noise Reduction](https://github.com/NeemaJ/BirdCLEF/blob/7eb47f058ba28db13be4d55a0ff60d73248a5fe3/PreProcessingCode/NoiseReduction.py)
- [HyperParameter Tuning](https://github.com/NeemaJ/BirdCLEF/tree/2e6f1df785ef8c3e69d5ad0e217cd13494272a99/hyperparam_tuning/bird_classification)
- [Train Small Audio data](https://github.com/NeemaJ/BirdCLEF/tree/109f21bd1a687658eb23c4d71ba4950ad002d64d/train_audio_smaller)



## Data Pipeline
![Data_Pipeline](https://github.com/user-attachments/assets/cd8f2e86-d8dd-42ee-b8ce-9639279e44f2)


##  Preprocessing
To divide the sound file into signal and noise parts, we first computed a spectrogram of the entire file using a short-time Fourier transform (STFT). For the signal, we applied binary filters to remove noise, created an indicator vector that marks the signal interval, and identified pixels from the spectrogram that were three times the row and column median. A similar process is used for noise. The threshold is slightly lower with 2.5 times the median. The signal and noise are then extracted, and placed into separate files. Their respective spectrograms are then computed. 
- signal/noise separation
- Chunk Division

## Data Augmentation 

## Results

## Improvments


