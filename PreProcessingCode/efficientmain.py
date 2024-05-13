
from CreatingSpectrograms import *
from RetreivingAudioFiles import *
from DataAugmentation import *
from NoiseReduction import *
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import time
import random
from CNN import *
import numpy as np
import os

#constants
path_to_created_specs = 'C:\\Users\\njrav\DS\\bird_chunked_specs'
path_to_created_augments = 'C:\\Users\\njrav\DS\\bird_augmented'
folder_path = 'C:\\Users\\njrav\DS\\train_one_audio'
freq_bins = 256
chunk_length = 512
noise_reduce_factor = 0.4
debug_mode = True

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



def save_spectrograms():
    #load each file name into the dictionary
    audio_dict = get_bird_audio_dict(folder_path)


    #loop through each folder
    for key in audio_dict:
        file_list = audio_dict[key]
	    
	#loop through each file
        for file in file_list:
			#calculate spectrogram
            tensor = get_audio_tensor(str(file))
            spectrogram = get_spectrogram(tensor)

			#separate noise/signal
			
			#min max scale
            mm_spec = min_max_scale_spectrogram(spectrogram).numpy()
			
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
            log_scale_spec = log_scale_spectrogram(spectrogram).numpy()

			#final signal and noise spec
            isolated_signal_spec = get_isolated_spectrogram(signal_dilated_indicator, log_scale_spec)
            isolated_noise_spec = get_isolated_spectrogram(noise_dilated_indicator_inverted, log_scale_spec)

			
			#X_train = np.empty((1,chunk_length,freq_bins))
			#y_train = np.empty(1)
	
	  
            chunked_specs = chunk_and_pad(isolated_signal_spec, chunk_length)
            chunked_noise = chunk_and_pad(isolated_noise_spec, chunk_length)

            #check to see if directory exists
            if not os.path.exists(path_to_created_specs):
                os.makedirs(path_to_created_specs)
            for index in range(len(chunked_specs)):
                if not os.path.exists(path_to_created_specs + f'\\{key}'):
                    os.makedirs(path_to_created_specs + f'\\{key}')
                np.save(path_to_created_specs + f'\\{key}\\{key}_signal_chunk{index}.npy', chunked_specs[index], allow_pickle=True)

            for index in range(len(chunked_noise)):
                np.save(path_to_created_specs + f'\\noise_chunk{index}.npy', chunked_noise[index], allow_pickle=True)

#TODO: don't store everything in memory all at once, only load numpy when we need. with getting sound files, only load numpy array when explicitly adding to another, choose based off file paths
def augment():
    spec_path_dict = dict()
    noise_path_list = list()
    bird_folders = list()
    for entry in os.scandir(path_to_created_specs):
        if entry.is_dir():
            bird_folders.append(entry)

    #testing
    '''
    for i in bird_folders:
        print(i)
    '''


    # Iterate over files in directory
    for bird_path in bird_folders:

        #bird_list is a generator
        bird_list = Path(bird_path).glob('**/*.npy')
        spec_path_dict[bird_path.name] = [f for f in bird_list]

    #get noise chunks paths
    noise_path_list = Path(path_to_created_specs).glob('*.npy')


    #testing
    '''
    for key in spec_path_dict:
        for thing in spec_path_dict[key]:
            print(thing)

    for thing in noise_path_list:
        print(thing)
    '''

    '''
    #load the npys
    spec_dict = dict()
    for key in spec_path_dict:
        spec_dict[key] = list()
        for file_path in spec_path_dict[key]:
            spec_dict[key].append(np.load(file_path, allow_pickle=True))


    noise_list = list()
    for file_path in noise_path_list:
        noise_list.append(np.load(file_path, allow_pickle=True))
    '''

    #do the augmenting
    for key in spec_path_dict:
        number_of_same_birds = 0
        chunked_spec_paths = spec_path_dict[key]

        for chunked_spec_path in chunked_spec_paths:
            #load in the npy
            chunked_spec = np.load(chunked_spec_path, allow_pickle=True)
            #time shift
            chunked_spec = time_shift(chunked_spec, random.randrange(chunk_length))

            #pitch shift
            chunked_spec = pitch_shift(chunked_spec, random.randrange(128))

            #add a random noise
            noise_to_add_path_1 = random.choice(noise_path_list)
            noise_to_add_1 = reduce_amplitude(np.load(noise_to_add_path_1, allow_pickle=True), noise_reduce_factor)
            noise_to_add_path_2 = random.choice(noise_path_list)
            noise_to_add_2 = reduce_amplitude(np.load(noise_to_add_path_2, allow_pickle=True), noise_reduce_factor)
            noise_to_add_path_3 = random.choice(noise_path_list)
            noise_to_add_3 = reduce_amplitude(np.load(noise_to_add_path_3, allow_pickle=True), noise_reduce_factor)

            if debug_mode:
                plot_abs_spectrogram(chunked_spec, 'chunked')
                plot_abs_spectrogram(noise_to_add_1, 'noise_to_add_1')
                plot_abs_spectrogram(noise_to_add_2, 'noise_to_add_2')
                plot_abs_spectrogram(noise_to_add_3, 'noise_to_add_3')
            
            chunked_spec = chunked_spec + noise_to_add_1 + noise_to_add_2 + noise_to_add_3

            if debug_mode:
                plot_abs_spectrogram(chunked_spec, 'chunked_spec_post')
            
            #save to folder
            if not os.path.exists(path_to_created_augments):
                os.makedirs(path_to_created_augments)
            path_to_current_bird = path_to_created_augments + f'\\{key}'
            if not os.path.exists(path_to_current_bird):
                os.makedirs(path_to_current_bird)
            np.save(path_to_current_bird + f'\\{key}_augment_{number_of_same_birds}', chunked_spec, allow_pickle=True)
            number_of_same_birds += 1

    

def main():
    st = time.time()

    #don't spend work trying to get spectrograms if we aready have them
    if not os.path.exists(path_to_created_specs):
        save_spectrograms()

    if not os.path.exists(path_to_created_augments):
        augment()
    
    

    print("Time it took:")
    print(time.time()-st)


if __name__ == "__main__":
    main()


