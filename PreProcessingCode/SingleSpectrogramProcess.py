

audio_dict = get_bird_audio_dict('/kaggle/input/birdclef-2024/train_audio')

audio_tensor = get_audio_tensor(str(audio_dict['brakit1'][1]))
spectrogram = get_spectrogram(audio_tensor)

fig, axes = plt.subplots(figsize=(12, 8))
plot_spectrogram(spectrogram.numpy(), axes)
