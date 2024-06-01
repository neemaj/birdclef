NEEMA_MAC = True
if not NEEMA_MAC:
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
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

#constants
NEEMA_MAC = True
RUN_NEEMA = True
model_count= 0
recheck_specs = False
recheck_augment = False
smaller_test = True

if NEEMA_MAC:
    path_to_created_specs = '/Volumes/Extreme SSD/DS/train_audio_smaller/bird_chunked_specs'
    path_to_created_augments = '/Volumes/Extreme SSD/DS/train_audio_smaller/bird_augmented'
    folder_path = '/Volumes/Extreme SSD/DS/train_audio_smaller/train_audio_smaller'
    best_model_path = f'/Volumes/Extreme SSD/DS/train_audio_smaller/model{model_count}.keras'
    pc = '/'
elif RUN_NEEMA:
    if smaller_test:
        path_to_created_specs = 'D:\\Downloads\\train_audio_smaller\\bird_chunked_specs'
        path_to_created_augments = 'D:\\Downloads\\train_audio_smaller\\bird_augmented'
        folder_path = 'D:\\Downloads\\train_audio_smaller\\train_audio_smaller'
        best_model_path = f'D:\\Downloads\\train_audio_smaller\\model{model_count}.keras'
    else:
        path_to_created_specs = 'D:\\DS\\bird_chunked_specs'
        path_to_created_augments = 'D:\\DS\\bird_augmented'
        folder_path = 'C:\\Users\\njrav\\DS\\BirdCLEF\\birdclef-2024\\train_audio'
        best_model_path = f'D:\\DS\\model{model_count}.keras'
    pc = '\\'
else:
    path_to_created_specs = '/Users/katiefrields/Desktop/BirdProject/BirdCLEF/PreProcessingCode/bird_chunked_specs'
    path_to_created_augments = '/Users/katiefrields/Desktop/BirdProject/BirdCLEF/PreProcessingCode/bird_augmented'
    folder_path = '/Users/katiefrields/Desktop/BirdProject/BirdCLEF/train_audio_smaller'
    best_model_path = f'/Users/katiefrields/Desktop/BirdProject/BirdCLEF/model{model_count}.keras'
    
    pc = '/'

freq_bins = 256
chunk_length = 512
noise_reduce_factor = 0.4
debug_mode = False

def get_path_label(folder_path, pc):
    '''
        Params:
            file_path: string path to folder containing all bird audio files
            
        Returns:
            dictionary where 
              key = array of audio file paths corresponding to that bird
              value = bird folder name as a string
    '''
    audio_dict = dict()
    bird_folders = os.listdir(Path(folder_path))
    
    ltrain = LabelEncoder()

    ltrain.fit(bird_folders)
    
    

    # Iterate over files in directory
    for bird_path in bird_folders:
        #bird_list is a generator
        bird_list = os.listdir(Path(folder_path + pc + bird_path))
        for f in bird_list:
            #audio_dict[f] = bird_path.name
            
            audio_dict[f] = ltrain.transform([bird_path])[0]
        
    return audio_dict


def save_bird_spectrograms(file, bird_name, file_id):
    path_to_created_spec = path_to_created_specs + f'{pc}{bird_name}{pc}{bird_name}_signal_chunk{file_id}-0.npy'
    path_to_created_noise = path_to_created_specs + f'{pc}noise_chunk{bird_name}-{file_id}-0.npy'
    if not os.path.exists(path_to_created_spec) or not os.path.exists(path_to_created_noise):
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
    
    
        if isolated_signal_spec.shape[0] != 0:
            chunked_specs = chunk_and_pad(isolated_signal_spec, chunk_length)
            chunked_noise = chunk_and_pad(isolated_noise_spec, chunk_length)
        
            for index in range(len(chunked_specs)):
                np.save(path_to_created_specs + f'{pc}{bird_name}{pc}{bird_name}_signal_chunk{file_id}-{index}.npy', chunked_specs[index], allow_pickle=True)
        
            for index in range(len(chunked_noise)):
                np.save(path_to_created_specs + f'{pc}noise_chunk{bird_name}-{file_id}-{index}.npy', chunked_noise[index], allow_pickle=True)



def save_spectrograms():
    #make the folder for all the spectrograms
    if not os.path.exists(path_to_created_specs):
        os.makedirs(path_to_created_specs)

    #load each file name into the dictionary
    audio_dict = get_bird_audio_dict(folder_path)

    #loop through all the birds folders
    for key in audio_dict:
        #makes the folder for that bird's new spectrograms
        if not os.path.exists(path_to_created_specs + f'{pc}{key}'):
            os.makedirs(path_to_created_specs + f'{pc}{key}')

        #gets the list of files for that bird
        file_list = audio_dict[key]
        tupled_files = [(file_path,key,index) for index, file_path in enumerate(file_list)]

        with Pool(max(1, os.cpu_count()- 1)) as p:
            p.starmap(save_bird_spectrograms, tupled_files)


        
def augment_bird_file(chunked_spec_path, key, file_id, noise_path_list):
    path_to_current_bird = path_to_created_augments + f'{pc}{key}'
    path_name = path_to_current_bird + f'{pc}{key}_augment_{file_id}'
    #20 percent chance of printing what bird we're on
    if debug_mode and random.randint(1,100) < 20:
        print(path_name)
    if not os.path.exists(path_name):
        
        #load in the npy
        chunked_spec = np.load(chunked_spec_path, allow_pickle=True)
    
        if chunked_spec.shape == (512,256):
            #time shift
            chunked_spec = time_shift(chunked_spec, random.randrange(chunk_length))
        
            #pitch shift
            chunked_spec = pitch_shift(chunked_spec, random.randrange(128))
    
            while True:
            
                #add a random noise
                noise_to_add_path_1 = random.choice(noise_path_list)
                noise_to_add_1 = reduce_amplitude(np.load(noise_to_add_path_1, allow_pickle=True), noise_reduce_factor)
                noise_to_add_path_2 = random.choice(noise_path_list)
                noise_to_add_2 = reduce_amplitude(np.load(noise_to_add_path_2, allow_pickle=True), noise_reduce_factor)
                noise_to_add_path_3 = random.choice(noise_path_list)
                noise_to_add_3 = reduce_amplitude(np.load(noise_to_add_path_3, allow_pickle=True), noise_reduce_factor)
    
                if noise_to_add_1.shape == (512,256) and noise_to_add_2.shape == (512,256) and noise_to_add_3.shape == (512,256):
                    break
        
            if debug_mode:
                plot_abs_spectrogram(chunked_spec, 'chunked')
                plot_abs_spectrogram(noise_to_add_1, 'noise_to_add_1')
                plot_abs_spectrogram(noise_to_add_2, 'noise_to_add_2')
                plot_abs_spectrogram(noise_to_add_3, 'noise_to_add_3')
            
        
    
            chunked_spec = chunked_spec + noise_to_add_1 + noise_to_add_2 + noise_to_add_3
    
                
        
            #if debug_mode:
                #plot_abs_spectrogram(chunked_spec, 'chunked_spec_post')
            
            #save to folder
            path_to_current_bird = path_to_created_augments + f'{pc}{key}'
        
            path_name = path_to_current_bird + f'{pc}{key}_augment_{file_id}'
            
            np.save(path_name, chunked_spec, allow_pickle=True)
    
            
#don't store everything in memory all at once, only load numpy when we need. with getting sound files, only load numpy array when explicitly adding to another, choose based off file paths
def augment():
    if not os.path.exists(path_to_created_augments):
        os.makedirs(path_to_created_augments)
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
        chunked_spec_paths = spec_path_dict[key]
        path_to_current_bird = path_to_created_augments + f'{pc}{key}'
        if not os.path.exists(path_to_current_bird):
            os.makedirs(path_to_current_bird)

        tupled_files = [(file_path,key,index, noise_path_list) for index, file_path in enumerate(chunked_spec_paths)]

        with Pool(max(1, os.cpu_count()- 1)) as p:
            p.starmap(augment_bird_file, tupled_files)


def main():
    
    st = time.time()
    # Set the number of intra-op parallelism threads
    #tf.config.threading.set_intra_op_parallelism_threads(2)

    # Set the number of inter-op parallelism threads
    #tf.config.threading.set_inter_op_parallelism_threads(2)

    #don't spend work trying to get spectrograms if we aready have them
    if recheck_specs or not os.path.exists(path_to_created_specs):
        spec_start_time = time.time()
        save_spectrograms()
        print("Time it took to make spectrograms:")
        print(f'{time.time()-spec_start_time}')
    else:
        print(f'{path_to_created_specs} already exists')

    if recheck_augment or not os.path.exists(path_to_created_augments):
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
   
    
   
        
        
    #finish all the getting the the spectrograms
    #we will have path to the augmented spectrograms
    #we have a folder of aug specs
    #make a dictionary that matches every path to its label
        
    #get a list of all of those spec paths

    #we will use train test split to split into spec paths for the validation and training data
    train_paths, valid_paths = train_test_split(file_path_list, test_size=0.10, random_state=42)

    X_train, X_valid = np.array(train_paths), np.array(valid_paths)
    # we will then feed the X_augs_train and label dictionary to a generator to make the training generator
    run_small_hp_model( train_paths, valid_paths, labels_dict, best_model_path)
    global model_count
    model_count+= 1
    # we will then feed the X_augs_validation and label dictionary  to a generator to make the validation generator 
        
        
    print("Train time:")
    print(time.time()-train_start_time)
    print("Total time it took:")
    print(time.time()-st)
    '''

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model('/Users/katiefrields/Desktop/BirdProject/BirdCLEF/model0.keras')

    # Show the model architecture
    new_model.summary()

    parsed = new_model.get_config()
    print(json.dumps(parsed, indent=4))
'''

    

if __name__ == "__main__":
    main()


