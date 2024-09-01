# Bird Audio-Call Identification 🐦

## Objective 
In this project, we created a convolutional neural network (CNN) to identify bird species in the Western Ghats, a biodiversity hotspot.
Our approach included:
- Implement preprocessing steps to handle to isolate and clean audio signal
- Employ multiprocessing, generators, and disk-writing to improve computation efficiency
- Utilize a CNN for identifying the species of origin for calls.


## Table Of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Data Augmentation](#data-augmentation)
- [Results](#results)<img width="1051" alt="Screen Shot 2024-08-18 at 3 11 06 PM" src="https://github.com/user-attachments/assets/083e3ae8-ddcb-4392-b510-15ba902db2e4">

- [Improvements](#improvments)

## Citation
Sprengel, E., Jaggi, M., Kilcher, Y., & Hofmann, T. (2016). Audio-Based Bird Species Identification using Deep Learning Techniques. Eidgenössische Technische Hochschule (ETH) Zürich, Zürich, Switzerland. Retrieved from ETH Zurich.

We implemented our approaches inspired by the techniques used in this paper "Audio-Based Bird Species Identification using Deep Learning Techniques" by Sprengel.

## Dataset
Our data was used from the BirdClef Kaggle competition. We had training data that had short recordings of bird calls.

Our primary issue was finding a method to convert audio into an image that could be processed by our CNN. To achieve this, we transformed the audio data into spectrograms while making sure to isolate the bird calls. We also standardized the spectrograms into the same size and amplitude scale. We accounted for background noise and enhanced our model by layering different types of background noise and applying pitch and time shifting to the audio. 

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
- Signal/Noise Separation
  
  To achieve this we:

  1) Created signal and noise masks

  2) Apply binary erosion and dilation to the spectrogram

  3) Observe how dilation refines signal presence and signal presence indicator
  <img width="1186" alt="Screen Shot 2024-08-14 at 4 00 07 PM" src="https://github.com/user-attachments/assets/33999301-4d4d-4587-b1ed-a993ce4c319f">
  
  <img width="1139" alt="Screen Shot 2024-08-14 at 4 00 28 PM" src="https://github.com/user-attachments/assets/b4db9350-26a1-4396-a7d3-6b75aed7746c">

- Creating Spectrograms

This code shows the audio waveforms in spectrograms, which can then be used as input to a convolutional neural network (CNN). Some functions generate spectrograms using Short-Time Fourier Transform (STFT), normalize the spectrogram data, and apply log scaling for better frequency representation. The code allows visualizing the spectrograms and isolating specific segments based on predefined criteria. The source code is inspired by TensorFlow's audio processing tutorial.

<img width="1051" alt="Screen Shot 2024-08-18 at 3 11 06 PM" src="https://github.com/user-attachments/assets/f8d972cf-b142-47dd-9a12-9c53d7d4f1a2">



<img width="1186" alt="Screen Shot 2024-08-14 at 4 00 07 PM" src="https://github.com/user-attachments/assets/c74b896e-995a-4892-a6ae-87a7e61731b2">


## CNN

To create our Convolutional Neural Network (CNN), we used custom data generators with multi-processing, this allowed for dynamic data loading and preprocessing. It was designed with multiple convolutional layers, and optimized through hyperparameter tuning to balance accuracy and performance. 

<img width="990" alt="Screen Shot 2024-08-29 at 2 56 42 PM" src="https://github.com/user-attachments/assets/9a21d53b-c459-486b-a92c-3b54a4ec2081">


<img width="1015" alt="Screen Shot 2024-08-29 at 2 57 27 PM" src="https://github.com/user-attachments/assets/a5bb9736-b0b6-43f4-9384-235cb8cea935">
<img width="1023" alt="Screen Shot 2024-08-29 at 2 57 13 PM" src="https://github.com/user-attachments/assets/4a0457cf-46e3-49ea-b4b5-95b658b6258d">
<img width="1016" alt="Screen Shot 2024-08-29 at 2 56 59 PM" src="https://github.com/user-attachments/assets/ca036971-5381-4461-8d5c-5f16693b2d0e">


## Data Augmentation 

We performed data augmentation on the spectrograms using four key methods:

1. Splititng the spectrogram into chunks of a specific size and adding pads to the end to check all chunks are equal size.

2. Shifting the spectrogram along the time axis by a specified amount, simulating a time delay.

3. Shifting the spectrogram along the frequency axis to simulate a change in pitch.

4. Lowering the amplitude of the spectrogram by a given factor to resemble quieter audio.

<img width="941" alt="Screen Shot 2024-08-23 at 4 13 03 PM" src="https://github.com/user-attachments/assets/81c37624-319b-4bb7-8045-e71db4957a1e">

<img width="996" alt="Screen Shot 2024-08-23 at 4 12 42 PM" src="https://github.com/user-attachments/assets/e5716fa6-636b-4fcc-bf24-7be5a84ca52f">



## Results

Our model achieved an accuracy of 78.12% on both the training and validation data, which we initially found surprising as the test accuracy for a different model configuration also yielded the same accuracy. Despite variations in the number of parameters across models, this consistency in accuracy suggests a stable performance across different model architectures.

<img width="839" alt="Screen Shot 2024-09-01 at 11 41 50 AM" src="https://github.com/user-attachments/assets/5a29ce60-785c-44c1-bb06-12237912d590">

<img width="702" alt="Screen Shot 2024-09-01 at 11 42 18 AM" src="https://github.com/user-attachments/assets/1319b209-0638-4775-baaa-58fd92caf23a">




## Improvments


