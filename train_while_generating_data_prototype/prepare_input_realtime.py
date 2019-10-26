#python prepare_input_mask.py B_format test 1 1 ['theta','MV'] 'whole'

import os
import sys
import time
import scipy.io
from scipy.io import wavfile
from scipy import signal
from scipy.stats import vonmises
import prepare_data
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from stats import RunningStats
from pre_white_embed import pre_white_embed
import h5py
import librosa
import create_mixtures
from train_RNN_mask import train #train_spatial_keras
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt

# audio parameters 
fs = 16000  
Wlength = 2048 #2048
window = 'hann'
window_size = Wlength#1024
hop_size = 513 #512
overlap = Wlength*3/4
fft_size = Wlength

# DOA parameters
n_softmax = 36
min_frames = 70 #80 for DNN
n_sources = 3
n_features = 3
n_channels = 2
n_freq = 1025
n_mix = 23000

def mag(x):
   
   x = 20 * np.log10(np.abs(x))
   return x

    
def angles_dist(P0, Gx, Gy):
 
    Y = (np.conj(P0)*Gy).real
    X = (np.conj(P0)*Gx).real
    theta = np.arctan2( Y,X )
    theta_deg = (np.degrees(theta).astype(int)+360)%360
    
    return theta_deg
    
def evaluate_IRM(X,source_index):
    
    IRM = np.abs(X[source_index,:,:,:] ) / np.sum( np.abs(X[:,:,:,:] ),axis=0 ) 
    
    return IRM
    
    
def fill_spectrogram_tensor(p0,px,py):
    
    # STFT
    [_, _, P0] = signal.stft(p0, fs, 'hann', Wlength, overlap, fft_size)
    [_, _, Gx] = signal.stft(px, fs, 'hann', Wlength, overlap, fft_size)
    [_, _, Gy] = signal.stft(py, fs, 'hann', Wlength, overlap, fft_size)
    
    # fill tensor
    X = np.zeros([P0.shape[0],min_frames,3], np.complex64)

    X[:,:,0] = P0[:,0:min_frames]
    X[:,:,1] = Gx[:,0:min_frames]
    X[:,:,2] = Gy[:,0:min_frames]

    return X
    
def prepare_input_mask(vel, p0):


    ### INPUT DATA GENERATION
                
    # create mixtures
    p0_mix = np.sum(p0,axis=0)
    vel_mix = np.sum(vel,axis=0)
                
    # pressure tensors
    X = fill_spectrogram_tensor(p0_mix, vel_mix[:,0], vel_mix[:,1])
                
    # angles distribution
    theta_deg = angles_dist(X[:,:,0], X[:,:,1], X[:,:,2])
        
    # initialize tensor for all sources
    X_all = np.zeros([n_sources,n_freq,min_frames,2],np.complex64)
        
    for source_index in range(0,n_sources):
            
        # sources pressures
        X_source = fill_spectrogram_tensor(p0[source_index,:], vel[source_index,:,0], vel[source_index,:,1])
        X_all[source_index,:,:,:] = X_source[:,:,1:3]

    # define ideal masks
    for source_index in range(0,n_sources):
                
        # define IRMs (Ideal Ratio Masks)
        y[source_index,:,:] = np.mean(evaluate_IRM(X_all, source_index),axis=2)

    # create features
    Xtrain = np.zeros([n_freq,min_frames,n_features])
    Xtrain[:,:,:,0] = theta_deg
    Xtrain[:,:,:,1] = mag(X[:,:,1])
    Xtrain[:,:,:,2] = mag(X[:,:,2])
    
    # labels
    ytrain0 = y[:,0,:,:]
    ytrain1 = y[:,1,:,:]
    ytrain2 = y[:,2,:,:]
    
    return Xtrain, ytrain0, ytrain1, ytrain2
    