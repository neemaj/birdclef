
from CreatingSpectrograms import *
from RetreivingAudioFiles import *
from DataAugmentation import *
from NoiseReduction import *
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import time
import random
from CNN import *
from BirdGenerator import *
import numpy as np
import os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

#constants
RUN_NEEMA = True

if RUN_NEEMA:
    path_to_created_specs = 'D:\\DS\\bird_chunked_specs'
    path_to_created_augments = 'D:\\DS\\bird_augmented'
    folder_path = 'C:\\Users\\njrav\DS\\train_audio_smaller'
    pc = '\\'
else:
    path_to_created_specs = '/Users/katiefrields/Desktop/BirdProject/BirdCLEF/PreProcessingCode/bird_chunked_specs'
    path_to_created_augments = '/Users/katiefrields/Desktop/BirdProject/BirdCLEF/PreProcessingCode/bird_augmented'
    folder_path = '/Users/katiefrields/Desktop/BirdProject/BirdCLEF/train_audio_smaller'
    pc = '/'

freq_bins = 256
chunk_length = 512
noise_reduce_factor = 0.4
debug_mode = True



def save_bird_spectrograms():
    

def save_spectrograms():
    if not os.path.exists(path_to_created_specs):
        os.makedirs(path_to_created_specs)
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

            for index in range(len(chunked_specs)):
                
                if not os.path.exists(path_to_created_specs + f'{pc}{key}'):
                    os.makedirs(path_to_created_specs + f'{pc}{key}')
                np.save(path_to_created_specs + f'{pc}{key}{pc}{key}_signal_chunk{index}.npy', chunked_specs[index], allow_pickle=True)

            for index in range(len(chunked_noise)):
                np.save(path_to_created_specs + f'{pc}noise_chunk{index}.npy', chunked_noise[index], allow_pickle=True)

#don't store everything in memory all at once, only load numpy when we need. with getting sound files, only load numpy array when explicitly adding to another, choose based off file paths
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
    noise_path_list = list(Path(path_to_created_specs).glob('*.npy'))


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
    #dictionary that maps every path (key) to its bird (value)
    

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

            #if debug_mode:
                #plot_abs_spectrogram(chunked_spec, 'chunked_spec_post')
            
            #save to folder
            if not os.path.exists(path_to_created_augments):
                os.makedirs(path_to_created_augments)
            path_to_current_bird = path_to_created_augments + f'{pc}{key}'
            if not os.path.exists(path_to_current_bird):
                os.makedirs(path_to_current_bird)

            path_name = path_to_current_bird + f'{pc}{key}_augment_{number_of_same_birds}'
            
            np.save(path_name, chunked_spec, allow_pickle=True)
            number_of_same_birds += 1


def main():
    st = time.time()

    #don't spend work trying to get spectrograms if we aready have them
    if not os.path.exists(path_to_created_specs):
        spec_start_time = time.time()
        save_spectrograms()
        print("Time it took to make spectrograms:")
        print(f'{time.time()-spec_start_time}')
    else:
        print(f'{path_to_created_specs} already exists')

    if not os.path.exists(path_to_created_augments):
        aug_start_time = time.time()
        augment()
        print("Time it took to make augments:")
        print(f'{time.time()-aug_start_time}')
    else:
        print(f'{path_to_created_augments} already exists')

    
    train_start_time = time.time()
    
    labels_dict = get_path_label(path_to_created_augments, pc)
    file_path_list = list()
    bird_folders = list()
    for entry in os.scandir(path_to_created_augments):
        if entry.is_dir():
            bird_folders.append(entry)
    for bird_path in bird_folders:
        file_path_list.extend(list(Path(bird_path).glob('**/*.npy')))
   
    
    #print(f'bird is {labels_dict[file_path_list[0]]}')
    #print(file_path_list[0:5])
        
        
    #finish all the getting the the spectrograms
    #we will have path to the augmented spectrograms
    #we have a folder of aug specs
    #make a dictionary that matches every path to its label
        
    #get a list of all of those spec paths

    #we will use train test split to split into spec paths for the validation and training data
    train_paths, valid_paths = train_test_split(file_path_list, test_size=0.25, random_state=42)

    X_train, X_valid = np.array(train_paths), np.array(valid_paths)
    # we will then feed the X_augs_train and label dictionary to a generator to make the training generator
    run_small_gen_model( train_paths, valid_paths, labels_dict)

    # we will then feed the X_augs_validation and label dictionary  to a generator to make the validation generator 
        

        
        
        
    print("Train time:")
    print(time.time()-train_start_time)
    print("Total time it took:")
    print(time.time()-st)


if __name__ == "__main__":
    main()


