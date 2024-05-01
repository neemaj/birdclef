from CreatingSpectrograms import *
from RetreivingAudioFiles import *
from DataAugmentation import *
from NoiseReduction import *
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import time

folder_path = 'C:\\Users\\njrav\DS\\train_audio_smaller'

def get_spectrogram_dict():
    '''
    Method used to get dictionary of spectrograms from folder_path
    '''
    audio_dict = get_bird_audio_dict(folder_path)
    spectrogram_dict = dict()

    #get the audio tensors, then get the spectrograms
    #loops through folders
    for key in audio_dict:
        spectrogram_dict[key] = list()
        file_list = audio_dict[key]
        for file in file_list:
            tensor = get_audio_tensor(str(file))
            spectrogram_dict[key].append(get_spectrogram(tensor))

    return spectrogram_dict


def separate_noise(spectrogram_dict):
    '''
    Method used to get noise and signal

        Returns:
            Dictionary of signal, and list of noise, as a tuple
    '''
    noise_list = list()
    signals_dict = dict()

    for key in spectrogram_dict:
         signals_dict[key] = list()
         spectrogram_list = spectrogram_dict[key]
         for spec in spectrogram_list:
              #min max scale
              mm_spec = min_max_scale_spectrogram(spec).numpy()


              #mask
              signal_spec, noise_spec = mask(mm_spec)

              #erode/dilate
              binary_signal_spec = apply_binary_erosion_and_dilation_spectrogram(signal_spec)
              signal_indicator = get_slices_indicator(binary_signal_spec)
              signal_dilated_indicator = apply_binary_dilation_indicator(signal_indicator)



              binary_noise_spec = apply_binary_erosion_and_dilation_spectrogram(noise_spec)
              noise_indicator = get_slices_indicator(binary_noise_spec)
              noise_dilated_indicator = apply_binary_dilation_indicator(noise_indicator)
              noise_dilated_indicator_inverted = np.invert(noise_dilated_indicator)

              #isolated
              log_scale_spec = log_scale_spectrogram(spec).numpy()
              isolated_signal_spec = get_isolated_spectrogram(signal_dilated_indicator, log_scale_spec)
              isolated_noise_spec = get_isolated_spectrogram(noise_dilated_indicator_inverted, log_scale_spec)

              #add to respective data structures
              noise_list.append(isolated_noise_spec)
              signals_dict[key].append(isolated_signal_spec)




    return signals_dict, noise_list
    
def augment(signals_dict, noise_list):
    '''
    Method used to chunk, augment the data, and add in noise

        Params:
            signals_dict: dictionary of signal spectrograms
            noise_list: list of noise spectrograms
        Returns:
            3d numpy array of all the spectrograms, 1d numpy array of all the labels 
    '''

     

def preprocess():
    '''
    Method used for preprocessing
    '''
    spectrogram_dict = get_spectrogram_dict()
    signals_dict, noise_list = separate_noise(spectrogram_dict)
    #data, labels = augment(signals_dict, noise_list)




    plot_abs_spectrogram(signals_dict['ashwoo2'][2], 'Signal Test')
    plot_abs_spectrogram(noise_list[10], 'Noise Test')
    



    '''
    fig1, axes1 = plt.subplots(figsize=(12,8))
    plot_spectrogram(spectrogram_dict['asikoe2'][0].numpy(), axes1)
    '''

    

    
    





def main():
        st = time.time()
        preprocess()
        et = time.time()
        print(et-st)
if __name__ == "__main__":
	main()
