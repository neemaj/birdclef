import tensorflow_io as tfio
from pathlib import Path
import tensorflow as tf

def get_bird_audio_dict(folder_path):
    '''
        Params:
            file_path: string path to folder containing all bird audio files
            
        Returns:
            dictionary where 
              key = bird folder name as a string
              value = array of audio file paths corresponding to that bird
    '''
    audio_dict = dict()
    bird_folders= Path(folder_path).glob("*")

    # Iterate over files in directory
    for bird_path in bird_folders:

        #bird_list is a generator
        bird_list = Path(bird_path).glob('**/*.ogg')
        audio_dict[bird_path.name] = [f for f in bird_list]
        
    return audio_dict


def get_path_label(folder_path):
    '''
        Params:
            file_path: string path to folder containing all bird audio files
            
        Returns:
            dictionary where 
              key = array of audio file paths corresponding to that bird
              value = bird folder name as a string
    '''
    audio_dict = dict()
    bird_folders= Path(folder_path).glob("*")

    # Iterate over files in directory
    for bird_path in bird_folders:

        #bird_list is a generator
        bird_list = Path(bird_path).glob('**/*.ogg')
        for f in bird_list:
            audio_dict[f] = bird_path.name
        
    return audio_dict


def get_audio_tensor(file_path):
    '''
        Params:
            file_path: string of path to the audio file

        Return:
            1D audio tensor
    '''
    audio = tfio.audio.AudioIOTensor(file_path)
    audio_tensor = tf.squeeze(audio[:], axis=[-1])
    return audio_tensor



