# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:21:42 2026
This script gives a spectrum and a spectrogram of your audio in tested formats mp3 or flac
It can also look for the similarity between two audios to see the degree of resemblance
between the two in time.  
@author: SM
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa 
from scipy.signal import spectrogram

class Soundspectre():
    def __init__(self, audiofile):
        '''
        audiofile with format as a string.
        It is loaded and a windowing filter is applied for further treatment.
        Parameters
        ----------
        audiofile : str
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.audiofile = audiofile
        self.data, self.orig_samplerate = self.load_audio()
        (
            self.audio_samples,
            self.audio_duration,
            self.time_period,
            self.freq_spacing,
            ) = self.fourier_params()
        self.window_data = self.window_filter('hanning', self.data,None)
        
    def load_audio(self):
        '''
        Loadfile gets audio from the current directory.
        If the directory is different should change for it.
        The loadfile function can handle audio input in another format, which it 
        will convert to string.

        Returns
        -------
        audiosourcedata : 
            it is audio data.
        sample_rate : TYPE
            rate at which, the audio is sampled.

        '''
        if type(self.audiofile) is str:
            self.filename = self.audiofile.strip().split('.')[0]
            audiostring = self.audiofile

        else:
            audiostring = str(self.audiofile)
            self.filename = self.audiofile.strip().split('.')[0]
        
        currentdir = os.getcwd()
        file_location = os.path.join(currentdir, audiostring)
        audiosourcedata, sample_rate = librosa.load(file_location, sr=44000)
        
        return audiosourcedata, sample_rate
    
    def fourier_params(self):
        '''
        Extract different useful parameters

        Returns
        -------
        audio_samplelength : int
            Length of audio data
        audio_duration : 
            Length of signal in seconds
        time_period : TYPE
            the spacing between sampled points in time
        freq_spacing : TYPE
            the frequency resolution possible 

        '''
        audio_samplelength = len(self.data)
        audio_duration = audio_samplelength/self.orig_samplerate
        time_period = audio_duration/audio_samplelength
        freq_spacing = 1/audio_duration
        
        return audio_samplelength, audio_duration, time_period, freq_spacing
        
    def window_filter(self, filtertype:str, data:None, size:None):
        '''
        Time windowing a signal to minimize oscillations or artifacts.
        Either pass the data or datasize to create filter specified
        Parameters
        ----------
        filtertype : str
            DESCRIPTION.
        data : None
            DESCRIPTION.
        size : None
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        filters = True
        while filters:
            if filtertype == 'hanning':
                if data is not None:
                    filtersize = len(data)
                    filterarr = np.hanning(filtersize)
                    windowed_data = filterarr*data
                    filters = False
                    break
                else:
                    filterarr = np.hanning(size)
                    filters = False
                    break
            else:
                print('this filter is not yet coded, using default hanning')
                filtertype = 'hanning'
                continue 
            
        if data is not None:
            return windowed_data
        else:
            return filterarr
        
    def fft_spectra(self, window_data,sample_length:None):  
        '''
        Take the FFT of the windowed signal.
        If sample_length is provided, fft frequencies are generated based on this length
        fftfreq : for a sampled signal of length NT having period T, 
        DFT makes a period reptition of this signal ie (x.dirac comb) * dirac comb
        so in time, signal is periodic in nT, the frequency is w_k = kw_N = k*2pi/NT
        Parameters
        ----------
        window_data : TYPE
            DESCRIPTION.
        sample_length : None
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        #frequency grid
        if self.audio_samples == len(window_data):
            self.frequency_components = np.fft.fftshift(np.fft.fftfreq(self.audio_samples, self.time_period))
            self.data_fft = np.fft.fftshift(np.fft.fft(window_data))
            
        else:
            if sample_length is not None:
                self.frequency_components = np.fft.fftshift(np.fft.fftfreq(sample_length, self.time_period))
                self.data_fft = np.fft.fftshift(np.fft.fft(window_data))
 
            else: 
                raise Exception('sample_length not specified')
                
        return self.data_fft, self.frequency_components
    
    def scipyspectrogram(self):
        '''
        Generate a spectrogram using scipy module

        Returns
        -------
        None.

        '''
        f, t, S = spectrogram(
             np.array(self.data),
             44000,
             window='hann',
             nperseg=2048,
             noverlap=512,
             nfft=2048,
             scaling='density',
             mode='magnitude'
         )
        
        S_db = 10 * np.log10(S + 1e-12)
        plt.imshow( S_db, cmap='magma',aspect='auto', origin='lower', 
                   extent=[0, self.audio_duration, 0, self.orig_samplerate])
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("Spectrogram (dB)")
        plt.ylim(0, 10000)
        plt.colorbar(label="Amplitude (dB)")
        plt.show()
        
    def zero_padded_data(self, sample_length:int ):
        '''
        When two signals are involved such as for cross-correlation
        Zeropad the signals to avoid effects coming due to wrapping effects ie 
        direct FT/FFT is a circular transform. 0, 1...-1

        Parameters
        ----------
        sample_length : int
            DESCRIPTION.

        Returns
        -------
        data_zeropad : TYPE
            DESCRIPTION.

        '''
        data_zeropad = np.zeros(sample_length)
        data_zeropad[0:self.audio_samples] = self.window_data
        return data_zeropad
        
    def crosscorrelation(self, S1fft, S2fft):
        '''
        Take FFTs of two signals as input. 
        FT(correlation) = FT1.FT2
        FT of a correlation is a product of FTs of each signal
        Parameters
        ----------
        S1fft : TYPE
            DESCRIPTION.
        S2fft : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        cross_cor = np.real(np.fft.ifft(np.fft.ifftshift(S1fft*np.conj(S2fft))))
        self.xcorr = np.fft.fftshift(cross_cor)
        return self.xcorr
    
    def xcorr_lag(self,sample_1size, sample_2size):
        '''
        In order to plot correlation, we need the delays between the two signals
        If there are 2N-1 points for each signal, the delay will be measured across those points.
        Delay between two points in time is given by an amount that depends on sample rate
        It is points*time_period

        Parameters
        ----------
        sample_1size : TYPE
            DESCRIPTION.
        sample_2size : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.lag = np.arange(-(sample_2size-1), sample_1size) 
        self.lag_seconds = self.lag/self.orig_samplerate
        
    def spectrogram(self):
        '''
        A manual spectrogram to compare with scipy.
        Spectrum of a window is taken and added into a list.
        To have a continuous data, the window is slided by slidepoints.
        Inorder to have y axis as spectrum for x axis time, list is inverted.

        Returns
        -------
        None.

        '''
        nfft_windowsize = 2048
        fftslidepoints = 512
        spectrogram_list = []
        sample_window = self.window_filter('hanning', None, nfft_windowsize)
   
        norm = np.sqrt(np.sum(sample_window**2))

        for i in range(0, self.audio_samples-nfft_windowsize, fftslidepoints):
            sample_data = self.data[i:i+nfft_windowsize]
            window_segment = sample_data*sample_window
            sample_fft = np.fft.rfft(window_segment)/norm
            spectrogram_list.append(np.abs(sample_fft))
        
        spectrogram_dat = np.array(spectrogram_list).T
        self.spectrogram = 10 * np.log10(spectrogram_dat + 1e-6)
        
    def plot_spectre(self, plottype:str):
        if plottype=='harmonics':
            xf_positive = self.frequency_components[self.audio_samples//2:] #-ve freq - 0 - posit freq
            yf_magnitude = abs(self.data_fft[self.audio_samples//2:])
            
            fig = plt.figure(figsize=(10, 8))
            # First Subplot (Linear)
            ax = fig.add_subplot(2,1,1)
            ax.plot(xf_positive, yf_magnitude)
            ax.set_title(f'{self.filename} (Spectrum - Linear)')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.set_xlim([0,15000])
            ax.grid(True)

            # Second Subplot (Logarithmic)
            ax2 = fig.add_subplot(2,1,2)
            ax2.plot(xf_positive, yf_magnitude) # Changed 'fig.plt' to 'ax2.plot'
            ax2.set_yscale('log') 
            ax2.set_xscale('log')
            ax2.set_title('(Spectrum - Log)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_xlim([0.01,15000])
            ax2.grid(True)
            plt.show()
            fig.savefig(f'{self.filename} spectrum.png')

        if plottype =='spectrogram':
             plt.figure(figsize=(12, 6))

             # origin='lower' puts low frequencies at the bottom
             plt.imshow(self.spectrogram, cmap = 'magma', aspect='auto', origin='lower', 
                        extent=[0, self.audio_duration, 0, self.orig_samplerate])

             plt.title(f'Spectra: {self.filename}')
             plt.ylabel('Frequency (Hz)')
             plt.xlabel('Time (seconds)')
             plt.colorbar(label='Intensity (dB)')
             plt.ylim(0, 10000)
             plt.xlim(0,self.audio_duration)
             plt.savefig(f'{self.filename} spectrogram.png')
             plt.show()
        if plottype =='xcorr':
            plt.figure(figsize=(12, 6))
            
            plt.plot(self.lag_seconds, self.xcorr)
            plt.xlabel("Lag (seconds)")
            plt.ylabel("Correlation")
            plt.title("Cross-Correlation")
            plt.grid(True)
            plt.show()
             
if __name__=='__main__':
    score1 = 'Howard Shore - Wedding Plans.flac'
    myscore1 = Soundspectre(score1)
    samples_1 = myscore1.audio_samples
    data1 = myscore1.data
    
    score2 = 'Metric - Eclipse [All Yours] (Video).mp3'
    myscore2 = Soundspectre(score2)
    samples_2 = myscore2.audio_samples
    data2 = myscore2.data
    samples_for_xcorr= samples_1+samples_2 - 1
    
    data_with_pad = myscore1.zero_padded_data(samples_for_xcorr)
    fft_with_pad,freq1 = myscore1.fft_spectra(data_with_pad,samples_for_xcorr)
 
    data_with_pad2 = myscore2.zero_padded_data(samples_for_xcorr)
    fft_with_pad2,freq2 = myscore2.fft_spectra(data_with_pad2,samples_for_xcorr)
    myscore2.plot_spectre('harmonics')
    
    norm1 = np.sqrt(np.sum(data1**2))
    norm2 = np.sqrt(np.sum(data2**2))
    cross_cor = myscore1.crosscorrelation(fft_with_pad/norm1, fft_with_pad2/norm2)
    myscore1.xcorr_lag(samples_1, samples_2)
    myscore1.plot_spectre('xcorr')
    
