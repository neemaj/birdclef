# Report
- Splitting the Spectrogram into Chunks and Padding
	- Intuitive explanation
		- Reason
			- Audio files are gonna be different lengths, we want them all to be uniform for the neural network, but still have workable content
				- The network will make one prediction per chunk, and return whatever label got the most chunks
			- Padding is needed on the last chunk because the audio file won't evenly divide
		- This is done for each sample file on its own
	- How to Calculate
		- Python Method
		```python
		add zero value to array to make the spectrogram array divisible by chunk_size
		num_chunks = ceiling of spectrogram shape / chunk_size
		chunks_array = []
		
		for i in num_chunks:
			append to chunks_array the spectrogram[i * chunk_size: (i+1) * chunk_size]
		```
	- Applications for similar projects
		- not really necessary?

- Data Augmentation
	- Time Shift
		- Intuitive Explanation
			- Bird calls can appear at any point in spectrogram, so this would better train model for that
		- Python Libraries
			- [numpy.roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) to shift over array on time axis. it will loop back around stuff that went overboard.
				- We can do this with random numbers
	- Pitch Shift
		- Intuitive Explanation
			- Our article didn't explain much on this, and also seemed to pitch shift uniformly
				- I don't this makes sense. other articles like [this](https://www.sciencedirect.com/science/article/pii/S1574954120300340) seemed to have random selections of pitch shifting
			- I believe it's just shaking up the data more
				- [This](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10249348/) said pitch shifting led to "better-quality sound data," which also doesn't make sense
					- This article was also not clear on if it was a uniform shift or not. probably was a uniform shift
		- How to do it
			- numpy.roll like timeshift but on the frequency axis
	- Combining Same Class Audio Files
		- About
			- The paper's writeup basically says this did nothing
			- I believe this means adding the amplitude values, then normalizing the result based on the max amplitude of the original files
		- Intuitive Explanation
			- This was to simulate multiple birds at once
			- Also claimed to improve convergence since it allowed the system to see multiple patterns at once
		- How to do it
			- make sure both audio files are same length
				- repeat the shorter one until they are same length
				- Functions
					- [numpy.tile](https://numpy.org/doc/stable/reference/generated/numpy.tile.html)
					-  Code at bottom

			- add values of 2 numpy arrays together
			- df = (df - dfmin)/(dfmax - dfmin) ([from here](https://www.projectpro.io/recipes/normalize-matrix-numpy))
	- Adding Noise
		- Intuitive Explanation
			- They didn't want the model to train on each file's own background noise because it should be independent of that
				- chose a random noise sample for each
			- really important, reduces overfitting
			- Instead of having this noise be in between the actual bird noises, they added it on top of the file like Combining Same Class Audio Files above
		- How to do it
			- Same as Combining Same Class Audio Files, but can also add multiple noise files on one signal file
			- repeat smaller file in time as necessary
			- > they mentioned a dampening factor?
				- so maybe add it by a lesser amount (multiply all those values by a constant)
			- can add different noises on the same sample and have those be different things we train on
		- Other Notes
			- I know other models also "[add noise](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/)" throughout the machine learning process, but this isn't specifically augmenting the data 

---
```python
    current_length = arr.shape[axis]
    num_repeats = length // current_length
    remainder = length % current_length
    repeated_arr = np.tile(arr, num_repeats)
    if remainder > 0:
        repeated_arr = np.concatenate((repeated_arr, arr.take(indices=range(remainder), axis=axis)), axis=axis)

    return repeated_arr
```