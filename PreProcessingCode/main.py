from CreatingSpectrograms import *
from RetreivingAudioFiles import *
from DataAugmentation import *
from NoiseReduction import *
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import time
import random

#constants
folder_path = 'C:\\Users\\njrav\DS\\train_one_audio'
chunk_length = 256
noise_reduce_factor = 0.4
debug_mode = False

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
    X_train = np.empty((1,256,513))
    y_train = np.empty(1)

    #first, handle noises
    chunked_noise = list()
    for noise in noise_list:
        chunked_noise.extend(chunk_and_pad(noise, chunk_length))


    
    for key in signals_dict:
        full_signals_list = signals_dict[key]

        for spec in full_signals_list:

            #chunk and pad
            chunked_specs = chunk_and_pad(spec, chunk_length)
            
            shifted_specs = list()
            for chunked_spec in chunked_specs:
                #time shift
                chunked_spec = time_shift(chunked_spec, random.randrange(chunk_length))

                #pitch shift
                chunked_spec = pitch_shift(chunked_spec, random.randrange(512))

                #add a random noise
                noise_to_add_1 = reduce_amplitude(random.choice(chunked_noise), noise_reduce_factor)
                noise_to_add_2 = reduce_amplitude(random.choice(chunked_noise), noise_reduce_factor)
                noise_to_add_3 = reduce_amplitude(random.choice(chunked_noise), noise_reduce_factor)

                if debug_mode:
                    plot_abs_spectrogram(chunked_spec, 'chunked')
                    plot_abs_spectrogram(noise_to_add_1, 'noise_to_add_1')
                    plot_abs_spectrogram(noise_to_add_2, 'noise_to_add_2')
                    plot_abs_spectrogram(noise_to_add_3, 'noise_to_add_3')
                
                chunked_spec = chunked_spec + noise_to_add_1 + noise_to_add_2 + noise_to_add_3

                if debug_mode:
                    plot_abs_spectrogram(chunked_spec, 'chunked_spec_post')
                
                #add as training data
                chunked_spec = np.array([chunked_spec])
                X_train = np.append(X_train, chunked_spec, axis=0)
                y_train = np.append(y_train, key)
            

    return (np.delete(X_train, (0), axis=0), np.delete(y_train, 0))
    

    
     

def preprocess():
    '''
    Method used for preprocessing, returns X_train and y_train
    '''
    spectrogram_dict = get_spectrogram_dict()
    signals_dict, noise_list = separate_noise(spectrogram_dict)
    X_train, y_train = augment(signals_dict, noise_list)

    #testing
    if debug_mode:
        print(np.shape(X_train))
        print(np.shape(y_train))
        plot_abs_spectrogram(X_train[2], 'Final Test')
        print(y_train[2])

    return X_train, y_train


    #plot_abs_spectrogram(signals_dict['ashwoo2'][2], 'Signal Test')
    #plot_abs_spectrogram(noise_list[10], 'Noise Test')
    



    '''
    fig1, axes1 = plt.subplots(figsize=(12,8))
    plot_spectrogram(spectrogram_dict['asikoe2'][0].numpy(), axes1)
    '''

    

    
    





def main():
    st = time.time()
    X_train, y_train = preprocess()
    et = time.time()

    if debug_mode:
        print("Time it took:")
        print(et-st)
if __name__ == "__main__":
	main()
