# Spectrum of an audio
`soundspectre.py` is a simple python script intended to look at the spectrum of an audio.
Tested formats: .mp3, .flac
@author: SM

## General architecture 

### Python libraries 
- os, numpy, matplotlib, librosa, scipy

### Implementation
- Soundspectre is a class method that takes an input of the form filename.format of an audio file as a string
- The class method contains functions that allow to take FFT spectrum of the signal, spectrogram and crosscorrelation
- Output is generated as .png files and saved in the same folder as the script
