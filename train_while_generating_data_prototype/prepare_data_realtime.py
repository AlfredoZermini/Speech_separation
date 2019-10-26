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
n_time = 56866
n_time1 = 40000

# DOA parameters
n_sources = 3
n_utt = 10 #60 train #20 test
n_mix = 23000
  
B_format_folder = os.getcwd()

    
def prepare_data_train(n_sources_string, task):
    
    # folders
    brir_path = '/vol/vssp/mightywings/BRIRs_S3a'
    
    # generate list of TIMIT speakers
    list_save = []
    list, min_length = generate_list(task)
    
    # clone of the original list when all the elements are deleted
    list_save = list[:]

    # define tensors
    vel_tensor = np.zeros([n_sources,min_length,2])
    
    # initialize lists 
    angles_all_list = []
    angle1_past_list = [] 
    angle2_past_list = [] 
    angle2_list = [] 
    angle3_list = []
    #angles_list = [-135, -110, -90, -60, -30, 0, 30, 60, 90, 110, 135, 180]
    angles_list = [-135, -110, -90, -60]

    # time lenghts
    brir_len = len( librosa.core.resample( sf.read( os.path.join(brir_path, 'B-format_Ls_' + str(0) +'.wav' ) )[0][:,0], 48000, fs) )
        
    # loop over angle1
    for angle1 in angles_list:

        # fill angle1 past list
        angle1_past_list.append(angle1)
        angle2_past_list = []
        
        # load brir1
        brir1 = sf.read( os.path.join(brir_path, 'B-format_Ls_' + str(angle1) +'.wav' ) )
        brir1_res = np.zeros([brir_len,4])
        
        # resample to fs
        for ch in range(0,4):
            brir1_res[:,ch] =  librosa.core.resample( brir1[0][:,ch] , 48000, fs)
            
        # remove angle1 list from angle2 list
        angle2_list = [item for item in angles_list if item not in angle1_past_list]

        # loop over angle2
        for angle2 in angle2_list:
            
            # fill angle2 past list
            angle2_past_list.append(angle2)
            
            # load brir2
            brir2 = sf.read( os.path.join(brir_path, 'B-format_Ls_' + str(angle2) +'.wav' ) )
            brir2_res = np.zeros([brir_len,4])

            # resample to fs
            for ch in range(0,4):
                brir2_res[:,ch] =  librosa.core.resample( brir2[0][:,ch] , 48000, fs)
        
            # remove angle2 list from angle3 list
            angle3_list = [item for item in angle2_list if item not in angle2_past_list]

            # loop over angle3
            for angle3 in angle3_list: 

                # load bri3
                #print os.path.join(brir_path, 'B-format_Ls' + str(angle3) +'.wav' )
                brir3 = sf.read( os.path.join(brir_path, 'B-format_Ls_' + str(angle3) +'.wav' ) )                
                brir3_res = np.zeros([brir_len,4])

                # resample to fs
                for ch in range(0,4):
                    brir3_res[:,ch] =  librosa.core.resample( brir3[0][:,ch] , 48000, fs)

                # angles
                angles_all_list.append([angle1,angle2,angle3])
                
                print "%s  %s  %s" % (angle1, angle2, angle3)
                
                # initialize lists and vectors
                path_list = []
                files_list = []
                angle_list = [angle1,angle2,angle3]
                index_list = 0
           
                ### create mixtures
                # iterate over utterances mixtures
                for utt in range(0,n_utt):
                    
                    # loop on sources
                    for index in range(0,n_sources):

                        # restore full list if empty
                        if list == []:
                            list = list_save[:]
                        
                        # select random speech and remove it from list
                        speech_path = random.choice(list)
                        speech_sample =  sf.read(speech_path)[0]
                        list.remove(speech_path)

                        # assign pressures
                        if index == 0:
                            brir_res = brir1_res
                        elif index == 1:
                            brir_res = brir2_res
                        elif index == 2:
                            brir_res = brir3_res
                        
                        # define pressures
                        p0 = signal.convolve(  brir_res[:,0], speech_sample)[0:min_length]
                        vel_0 = signal.convolve(  brir_res[:,1], speech_sample)[0:min_length]
                        vel_1 = signal.convolve(  brir_res[:,2], speech_sample)[0:min_length]
                   
                        # fill tensors with individual sources vectors
                        p0 = p0/np.max(p0)
                        vel_tensor[:,:,:,0] = vel_0/np.max(vel_0)
                        vel_tensor[:,:,:,1] = vel_1/np.max(vel_1)
                        
                        prepare_input_realtime(vel_tensor, p0)
                        
                        index_list += 1
                        
                        
def train() 
    DNN.fit_generator(generator(Xtrain, ytrain0, ytrain1, ytrain2, batch_size, n_masks), samples_per_epoch=n_samples/batch_size, nb_epoch=n_epochs)
                        
if __name__ == '__main__':
    
    if sys.argv[1] != "" and sys.argv[2].lower() != '' and sys.argv[3] in ['1'] and set(['MV','theta']).issuperset(set(sys.argv[4].replace("[","").replace("]","").split(","))):
            
        print sys.argv
        DNN_name = sys.argv[1]
        task = sys.argv[2].lower()
        Train_Room = sys.argv[3]
        features_name = sys.argv[4]
        post_suffix = sys.argv[5]
        DNN_suffix = sys.argv[6]

        project_path = os.path.join('/vol/vssp/mightywings/', DNN_name)

        if project_path.startswith('DNN_'):
            project_path = project_path[len('DNN_'):]

        if post_suffix != '': 
            post_suffix = '_' + post_suffix        

        if DNN_suffix != '':
            DNN_suffix = '_' + DNN_suffix
            
        # number of sources
        if n_sources == 2:
           n_sources_string = 'Two'
            
        elif n_sources == 3:
            n_sources_string = 'Three'

        elif n_sources == 4:
            n_sources_string = 'Four'
  
        # get dataset name
        suffix =  ( (features_name.replace(",","_") ).replace("[","") ).replace("]","")

        # DNN folder
        DNN_name = os.path.join(project_path, 'Results/Room' + Train_Room, 'DNN_models_' + suffix + '_' + task + post_suffix + DNN_suffix + '.h5')

        main_mask()

    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()