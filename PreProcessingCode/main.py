from CreatingSpectrograms import *
from RetreivingAudioFiles import *
from DataAugmentation import *
from ThresholdMasking import *
import tensorflow_io as tfio
import matplotlib.pyplot as plt

folder_path = 'INSERT FOLDER PATH HERE CONTAINING ALL BIRD AUDIO FILES'

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


def preprocess():
    '''
    Method used for preprocessing
    '''
    spectrogram_dict = get_spectrogram_dict()

    '''
    fig1, axes1 = plt.subplots(figsize=(12,8))
    plot_spectrogram(spectrogram_dict['asikoe2'][0].numpy(), axes1)
    '''

    

    
    





def main():
        preprocess()

if __name__ == "__main__":
	main()
