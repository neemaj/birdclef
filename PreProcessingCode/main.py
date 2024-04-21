import CreatingSpectrograms
import RetreivingAudioFiles
import tensorflow_io as tfio



def testing():
        '''
        Method used for testing functions
        '''
        tfio.audio.AudioIOTensor('Users/neema/XC760790.ogg')
        #audio_tensor = RetreivingAudioFiles.get_audio_tensor('Users/neema/XC760790.ogg')
        #CreatingSpectrograms.get_spectrogram()

def main():
        testing()

if __name__ == "__main__":
	main()
