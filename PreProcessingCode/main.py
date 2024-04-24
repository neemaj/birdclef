import CreatingSpectrograms
import RetreivingAudioFiles
import DataAugmentation
import ThresholdMasking
import tensorflow_io as tfio

folder_path = 'INSERT FOLDER PATH HERE CONTAINING ALL BIRD AUDIO FILES'




def preprocess():
    '''
    Method used for preprocessing
    '''
    audio_dict = RetreivingAudioFiles.get_bird_audio_dict(folder_path)
    audio_tensor_dict = dict()

    for key in audio_dict:
        audio_tensor_dict[key] = list()
        file_list = audio_dict[key]
        for file in file_list:
            audio_tensor_dict[key].append(RetreivingAudioFiles.get_audio_tensor(file))


def main():
        preprocess()

if __name__ == "__main__":
	main()
