import os
import sys
import time
import scipy.io
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from stats import RunningStats
import h5py
import librosa
import random
from random import randint
from operator import sub
from generate_list import generate_list

# audio parameters 
fs = 16000  
Wlength = 2048 #2048
window = 'hann'
window_size = Wlength#1024
hop_size = 1025 #512
overlap = Wlength*3/4
fft_size = Wlength
n_time = 35021#56866

# DOA parameters
n_sources = 3
n_channels = 3
n_doas = 36
n_doas_mix = 9
n_utt = 1 #2500 train,3 sources #20 test
n_mix = n_utt*n_doas_mix #22500
lock = 1
  
B_format_folder = os.getcwd()

    
def prepare_data(n_sources_string, task, Room):
    
    # folders
    brir_path = '/vol/vssp/mightywings/B_format_RIRs_Alfredo_S3A/Room' +Room
    
    # generate list of TIMIT speakers
    list_save = []
    list, min_length = generate_list(task)
    #list = list[0:6] #!!!
    
    # clone of the original list when all the elements are deleted
    list_save = list[:]
    
    # define tensors
    vel_tensor = np.zeros([n_mix,n_sources,n_time,2])
    p0_tensor = np.zeros([n_mix,n_sources,n_time])
    label_tensor = np.zeros([n_mix,n_sources])
        
    # initialize indices
    mix_index = 0
    index_utt = 0
    
    # initialize lists 
    angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ] 
    #angles_list = [ [180,190,200,210], [180,200,220,240], [180,210,240,270], [180,220,260,300], [180,230,280,330],  [180,240,300,0], [180,250,320,30], [180,260,340,60], [180,270,0,90]  ]

    # time lenghts
    brirs = scipy.io.loadmat(os.path.join(brir_path, 'B_format_RIRs_12BB01_Alfredo_S3A') )['rirs_final']
    brirs_len = len(librosa.core.resample( brirs[0,0,:] , 48000, fs))
    
    # resample to 16kHz
    brirs_resample= np.zeros([n_doas,n_channels,brirs_len])
    for ch in range(0,n_channels):
        brirs_resample[:,ch,:] =  librosa.core.resample( brirs[:,ch,:] , 48000, fs)
    
    del brirs
    
    # loop over angles
    for angles in angles_list:
        
        angle0 = angles[0]
        angle1 = angles[1]
        angle2 = angles[2]
        angle3 = angles[3]

        print "%s  %s  %s  %s" % (angle0, angle1, angle2, angle3)
                
        # initialize lists and vectors
        path_list = []
        files_list = []
        if n_sources_string == 'Three':
            angle_list = [str(angle0), str(angle1), str(angle2)]
        elif n_sources_string == 'Four':
            angle_list = [str(angle0), str(angle1), str(angle2), str(angle3)]
        index_list = 0
           
           
        if task == 'test':
            p0_mat = np.zeros([n_time,1,n_utt,n_sources])
            vel_mat = np.zeros([n_time,2,n_utt,n_sources])
            mixind_mat = np.zeros([n_sources,n_utt])    
            utt_mat = np.zeros([n_time,n_utt*n_sources])
            mix_path_mat = os.path.join('/vol/vssp/mightywings/B_format/matlab_code/mat_files/BformatMixtures/',n_sources_string + 'Sources')
           
        ### create mixtures
        # iterate over utterances mixtures
        for index_utt in range(0,n_utt):
                    
            # loop on sources
            for index in range(0,n_sources):

                # restore full list if empty
                if list == []:
                    list = list_save[:]
                        
                # select random speech and remove it from list
                speech_path = random.choice(list)
                speech_sample =  sf.read(speech_path)[0][0:min_length]
                list.remove(speech_path)

                # assign pressures
                if index == 0:
                    brir_res = brirs_resample[angle0/10,:,:]
                elif index == 1:
                    brir_res = brirs_resample[angle1/10,:,:]
                elif index == 2:
                    brir_res = brirs_resample[angle2/10,:,:]
                elif index == 3:
                    brir_res = brirs_resample[angle3/10,:,:]
                        
                # define pressures
                
                p0 = signal.convolve(  brir_res[0,:], speech_sample)[0:min_length]
                vel_0 = signal.convolve(  brir_res[1,:], speech_sample) [0:min_length]
                vel_1 = signal.convolve(  brir_res[2,:], speech_sample)[0:min_length]
                '''
                target_path = '/user/cvssppgr/az00147/Documents/Python/B_format/p0.wav'
                wavfile.write(target_path,fs,p0/np.max(np.abs(p0)))
                target_path = '/user/cvssppgr/az00147/Documents/Python/B_format/vel_0.wav'
                wavfile.write(target_path,fs,vel_0/np.max(np.abs(p0)) )
                target_path = '/user/cvssppgr/az00147/Documents/Python/B_format/speech_sample.wav'
                wavfile.write(target_path,fs,speech_sample/np.max(np.abs(p0)))
                cocco0
                '''

                # fill tensors with individual sources vectors
                p0_tensor[mix_index,index,:] = p0/np.max(np.abs(p0))
                vel_tensor[mix_index,index,:,0] = vel_0/np.max(np.abs(p0))
                vel_tensor[mix_index,index,:,1] = vel_1/np.max(np.abs(p0))
                label_tensor[mix_index,index] = int(angle_list[index])
                
                ### DATA CREATION FOR MATLAB CODE (baseline)
                if task == 'test':
                    
                    # b-format channels
                    p0_mat[:,0,index_utt,index] = p0/np.max(np.abs(p0))
                    vel_mat[:,0,index_utt,index] = vel_0/np.max(np.abs(p0))
                    vel_mat[:,1,index_utt,index] = vel_1/np.max(np.abs(p0))
                    
                    # mixing indices
                    mixind_mat[index,index_utt] = n_sources*index_utt+index+1
                    
                    # original utterances
                    utt_mat[:,index_list] = p0/np.max(np.abs(p0))
                        
                    index_list += 1

            # increase mixture index
            mix_index = mix_index+1
            
        ### DATA CREATION FOR MATLAB CODE (baseline)
        if task == 'test' and lock == 0:
            
            # create mixtures
            p0_mix_mat = np.sum(p0_mat,axis=3)
            vel_mix_mat = np.sum(vel_mat,axis=3)
            
            #angles folders
            angles_folder = os.path.join(mix_path_mat, '_'.join(angle_list)  )
            if not os.path.exists(angles_folder):
                os.makedirs(angles_folder) 
            scipy.io.savemat(os.path.join(angles_folder,'p03_humm.mat'), {'p03_humm':p0_mix_mat})
            scipy.io.savemat(os.path.join(angles_folder,'Vel3_humm.mat'), {'Vel3_humm':vel_mix_mat})
            scipy.io.savemat(os.path.join(angles_folder,'Mixing_ind_humm.mat'), {'Mixing_ind_humm':mixind_mat})
            scipy.io.savemat(os.path.join(angles_folder,'utterances_bformat.mat'), {'utterances_bformat':utt_mat})
            
    # delete empty elements
    p0_tensor = p0_tensor[0:mix_index,:,:]
    vel_tensor = vel_tensor[0:mix_index,:,:,:]
    label_tensor = label_tensor[0:mix_index,:]
    
    return vel_tensor, p0_tensor, label_tensor
