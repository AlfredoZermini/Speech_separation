#python prepare_input_mask.py B_format train 12BB01 12BB01 ['theta','MV'] ''
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
import math

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
n_doas_mix = 9
n_utt = 1 #2500 train,3 sources #20 test
n_mix = n_utt*n_doas_mix #22500

def mag(x):
   
   x = 20 * np.log10(np.abs(x))
   return x

    
def angles_dist(P0, Gx, Gy):
 
    Y = (np.conj(P0)*Gy).real
    X = (np.conj(P0)*Gx).real
    theta =  np.arctan2( Y,X )

    '''
    for t in range(0, theta.shape[1]):
        for f in range(0,theta.shape[0]):
            if (Y[f,t] < 0 and X[f,t] > 0) or (Y[f,t] < 0 and X[f,t] < 0):
                theta[f,t] = theta[f,t]+2*np.pi
                
            #theta[theta<0] = theta+2*np.pi
    '''
    
    #theta_deg = (np.degrees(theta))#.astype(int))
    
    #theta_deg = (theta_deg + 360) % 360
    theta_deg = (np.degrees(theta).astype(int)+360)%360
    
    return theta_deg
    
def evaluate_IRM(X,source_index):
    
    IRM = np.abs(X[source_index,:,:,:] ) / np.sum( np.abs(X[:,:,:,:] ),axis=0 ) 
    
    return IRM
    
def evaluate_IBM(X,source_index):
    
    # define IBM
    IBM = np.zeros([X.shape[1],X.shape[2],2])
    
    # fill TF bins
    for ch in range(0,2):
        for f in range(0,X.shape[1]):
            for t in range(0,X.shape[2]):
            
                max = 0
            
                for idx in range(0,len(X)):
                
                    if np.abs(X[idx,f,t,ch]) > max:
                        max = np.abs(X[idx,f,t,ch])
    
                if np.abs(X[source_index,f,t,ch]) >= max:
                    IBM[f,t,ch] = 1
    
    return IBM
    
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
    
def prepare_input_mask():

    
    # define mixture spectrograms tensor
    if task == 'test':
       print "%s - Creating spectrograms tensors." % time.ctime()
       Xmix_tensor = np.zeros((n_mix, n_freq, min_frames,n_channels), dtype=np.complex64)
       pressure_tensor = np.zeros([n_mix,n_sources,n_freq,min_frames,3], np.complex64)
    
    # define tensors and lists
    theta_tensor = np.zeros([n_mix,n_freq,min_frames])
    Gx_tensor = np.zeros([n_mix,n_freq,min_frames])
    Gy_tensor = np.zeros([n_mix,n_freq,min_frames])
    IRM_tensor = np.zeros([n_mix,n_sources,n_freq,min_frames])
    #IBM_tensor = np.zeros([n_mix,n_sources,n_freq,min_frames])
    
    # running mean and std
    runstats = RunningStats(n_freq, np.float64)
    stats = []


    ### INPUT DATA GENERATION

    # create mixtures
    print "%s - Creating data." % (time.ctime())

    [vel_tensor, p0_tensor, raw_label_tensor] = prepare_data.prepare_data(n_sources_string,task,Train_Room )
    
    if task == 'train':
        del raw_label_tensor

    print "%s - Data created!." % (time.ctime())

    print "%s - Filling data tensors." % (time.ctime())

    for mix_index in range(0,p0_tensor.shape[0]):
        
        #print raw_label_tensor[mix_index,:]
           
        # pressure paths
        p0 = p0_tensor[mix_index,:,:]
        vel = vel_tensor[mix_index,:,:,:]
                
        # create mixtures
        p0_mix = np.sum(p0,axis=0)
        vel_mix = np.sum(vel,axis=0)
                
        # pressure tensors
        X = fill_spectrogram_tensor(p0_mix, vel_mix[:,0], vel_mix[:,1])
        del p0_mix, vel_mix
                
        # angles distribution
        theta_deg = angles_dist(X[:,:,0], X[:,:,1], X[:,:,2])
        
        # initialize tensor for all sources
        X_all = np.zeros([n_sources,n_freq,min_frames,2],np.complex64)
        
        for source_index in range(0,n_sources):
            
            # sources pressures
            X_source = fill_spectrogram_tensor(p0[source_index,:], vel[source_index,:,0], vel[source_index,:,1])
            X_all[source_index,:,:,:] = X_source[:,:,1:3]
            
            if task == 'test':
                pressure_tensor[mix_index,source_index,:,:,:] = X_source
                '''
                plt.imshow(np.abs(pressure_tensor[mix_index,0,:,:,1]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
                clb = plt.colorbar()
                plt.gca().invert_yaxis()
                plt.savefig(os.path.join(figures_path, 'pressure_tensor' + str(mix_index) + '_' + str(int(raw_label_tensor[mix_index,0])) ))
                plt.clf()
                plt.show()

                plt.imshow(np.abs(pressure_tensor[mix_index,1,:,:,1]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
                clb = plt.colorbar()
                plt.gca().invert_yaxis()
                plt.savefig(os.path.join(figures_path, 'pressure_tensor' + str(mix_index) + '_' + str(int(raw_label_tensor[mix_index,1])) ))
                plt.clf()
                plt.show()
                
                plt.imshow(np.abs(pressure_tensor[mix_index,2,:,:,1]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
                clb = plt.colorbar()
                plt.gca().invert_yaxis()
                plt.savefig(os.path.join(figures_path, 'pressure_tensor' + str(mix_index) + '_' + str(int(raw_label_tensor[mix_index,2])) ))
                plt.clf()
                plt.show()
                '''
        # define ideal masks
        for source_index in range(0,n_sources):
                
            # define IRMs (Ideal Ratio Masks)
            IRM_tensor[mix_index,source_index,:,:] = np.mean(evaluate_IRM(X_all, source_index),axis=2)
            
            plt.imshow(IRM_tensor[mix_index,source_index,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            clb = plt.colorbar()
            plt.tick_params(labelsize=18)
            plt.xlabel(r'$\omega$', fontsize=25)
            plt.ylabel('f', fontsize=25)
            clb.ax.tick_params(labelsize=18) 
            plt.title('IRM source ' + str(source_index), fontsize=30)
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(figures_path, 'IRM' + str(mix_index) + '_' + str(source_index)))
            plt.clf()
            plt.show()
            
            # define IBMs (Ideal Binary Masks)
            #IBM_tensor[mix_index,source_index,:,:] = np.mean(evaluate_IBM(X_all, source_index),axis=2)
        
        # fill tensor
        theta_tensor[mix_index,:,:] = theta_deg
        print mix_index*10

        
        Gx_tensor[mix_index,:,:] = mag(X[:,:,1])
        Gy_tensor[mix_index,:,:] = mag(X[:,:,2])

        
        fig,ax = plt.subplots(1,figsize=(14,14))
        n, bins, patches = plt.hist(theta_deg.flatten(), bins=72, facecolor='blue', alpha=0.5, edgecolor='black', linewidth=1.2)
        plt.tick_params(labelsize=28)
        plt.xlabel(r'$\theta$ (degrees)', fontsize=36)
        plt.ylabel(r'$\theta$ count', fontsize=36)
        plt.title('Distribution of angles', fontsize=42)
        plt.show()
        plt.savefig(os.path.join(figures_path, 'theta_deg' + str(mix_index)),bbox_inches='tight')
        plt.clf()
        
        plt.imshow(theta_tensor[mix_index,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Distribution  of angles', fontsize=42)
        plt.show()
        plt.savefig(os.path.join(figures_path, 'theta_tensor' + str(mix_index)  ), bbox_inches='tight')
        plt.clf()
        plt.show()
        
        plt.imshow(Gx_tensor[mix_index,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Spectrogram: ' + '$G_x$', fontsize=42)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(figures_path, 'Gx' + str(mix_index)  ),bbox_inches='tight' )
        plt.clf()
        plt.show()
        
        plt.imshow(Gy_tensor[mix_index,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Spectrogram: ' + '$G_y$', fontsize=42)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(figures_path, 'Gy' + str(mix_index)  ),bbox_inches='tight')
        plt.clf()
        plt.show()
        
        
        # update stats
        runstats.update(np.hstack([theta_tensor[mix_index,:,:],Gx_tensor[mix_index,:,:],Gy_tensor[mix_index,:,:] ]).T )
            
        if task == 'test':
                
            # fill original mixtures spectrograms
            Xmix_tensor[mix_index,:,:,0] = X[:,:,1] #left
            Xmix_tensor[mix_index,:,:,1] = X[:,:,2] #right
            
        del X, X_all, X_source, theta_deg


    ### FINAL EVALUATION        
    
    # cut empty elements in tensor    
    theta_tensor = theta_tensor[0:p0_tensor.shape[0],:,:]
    IRM_tensor = IRM_tensor[0:p0_tensor.shape[0],:,:,:]
    #IBM_tensor = IBM_tensor[0:p0_tensor.shape[0],:,:,:]
    print IRM_tensor.shape
    
    
    if 'MV' in features_name:
        Gx_tensor = Gx_tensor[0:p0_tensor.shape[0],:,:]
        Gy_tensor = Gy_tensor[0:p0_tensor.shape[0],:,:]
    

    # cut spectrograms
    if task == 'test':
       raw_label_tensor = raw_label_tensor[0:p0_tensor.shape[0],:]
       Xmix_tensor = Xmix_tensor[0:p0_tensor.shape[0],:,:,:]
       pressure_tensor = pressure_tensor[0:p0_tensor.shape[0],:,:,:,:]

    # fill features tensor
    features_all = np.zeros([theta_tensor.shape[0],theta_tensor.shape[1],theta_tensor.shape[2],n_features])
    features_all[:,:,:,0] = theta_tensor
    features_all[:,:,:,1] = Gx_tensor
    features_all[:,:,:,2] = Gy_tensor

    del theta_tensor, Gx_tensor, Gy_tensor
    
    # use mean and variance
    if task == 'train':
        # fill stats list
        print "%s - Evaluating mean and std for normalization." % (time.ctime())

        mean = np.zeros([n_freq])
        std = np.zeros([n_freq])
        
        mean =  runstats.stats['mean']
        std = np.sqrt(runstats.stats['var'])
                
        print mean
        print std
        
        stats.append([mean,std])
        
        #print(np.mean(np.hstack([theta_tensor[mix_index,:,:],Gx_tensor[mix_index,:,:],Gy_tensor[mix_index,:,:] ]).T, axis=0))
        #print(np.sqrt(np.var(np.hstack([theta_tensor[mix_index,:,:],Gx_tensor[mix_index,:,:],Gy_tensor[mix_index,:,:] ]).T, axis=0, ddof=1)))
    
    elif task == 'test':
        # use train mean and std
        with h5py.File(save_train_data_path, 'r') as hf:
           stats = hf.get('stats')
           stats = np.array(stats)
        mean = stats[0][0]
        std = stats[0][1]  

    # normalization
    print "%s - Normalizing." % (time.ctime())
    for f in range(0,features_all.shape[1]):
        features_all[:,f,:,:] = (features_all[:,f,:,:]-mean[f])/std[f]
    
    
    # save data   
    print "%s - Saving %s data tensor." % (time.ctime(), task)

    if task == 'train':

        with h5py.File(save_train_data_path, 'w') as hf:
           hf.create_dataset('data', data=features_all)
           hf.create_dataset('IRM', data=IRM_tensor)
           #hf.create_dataset('IBM', data=IBM_tensor)
           hf.create_dataset('stats', data=stats)

    elif task == 'test':
        with h5py.File(save_test_data_path, 'w') as hf:
           hf.create_dataset('data', data=features_all)
           hf.create_dataset('spectrograms_mixture', data=Xmix_tensor)
           hf.create_dataset('spectrograms_original', data=pressure_tensor)
           hf.create_dataset('labels', data=raw_label_tensor)
        
    print "%s - Done!." % (time.ctime())
   
   
if __name__ == '__main__':
    
    if sys.argv[1] != "" and sys.argv[2].lower() != '' and sys.argv[3] in ['1','12BB01'] and sys.argv[4] in ['1','12BB01'] and set(['MV','theta']).issuperset(set(sys.argv[5].replace("[","").replace("]","").split(","))):
   
        print sys.argv
        DNN_name = sys.argv[1]
        task = sys.argv[2].lower()
        Train_Room = sys.argv[3]
        Test_Room = sys.argv[4]
        features_name = sys.argv[5]
        post_suffix = sys.argv[6]
       
        if post_suffix != '':
            post_suffix = '_' + post_suffix
      
        # get dataset name
        suffix =  ( (features_name.replace(",","_") ).replace("[","") ).replace("]","")
        
        # number of sources
        if n_sources == 1:
           n_sources_string = 'One'
        elif n_sources == 2:
           n_sources_string = 'Two'
            
        elif n_sources == 3:
            n_sources_string = 'Three'

        elif n_sources == 4:
            n_sources_string = 'Four'
            
        print n_sources_string
    
        # folders
        project_path = '/vol/vssp/mightywings'
        B_format_folder = os.getcwd()
        if task == 'train':
            figures_path = os.path.join(B_format_folder, 'Figures/train')
        elif task == 'test':
            figures_path = os.path.join(B_format_folder, 'Figures/test')
            
        # train and test paths
        save_TrainData_path = os.path.join(project_path, DNN_name, 'InputData/TrainData')
        save_train_room_path = os.path.join(save_TrainData_path, 'Room' + Train_Room)
        save_train_data_path = os.path.join(save_train_room_path, 'train' + '_'  +'TrainRoom' + Train_Room + '_' + n_sources_string + 'Sources' + post_suffix + '.h5')
        save_stats_data_path = os.path.join(save_train_room_path, 'stats' + '_'  +'TrainRoom' + Train_Room + '_' + n_sources_string + 'Sources' + post_suffix + '.h5')
           
        save_TestData_path = os.path.join(project_path, DNN_name, 'InputData/TestData')
        save_test_room_path = os.path.join(save_TestData_path, 'Room' + Train_Room)
        save_test_data_path = os.path.join(save_test_room_path, 'test' + '_'  +'TestRoom' + Train_Room + '_' + n_sources_string + 'Sources' + post_suffix + '.h5')
        
        if not os.path.exists(save_train_room_path):
            os.makedirs(save_train_room_path)
        if not os.path.exists(save_test_room_path):
            os.makedirs(save_test_room_path)    
        

        prepare_input_mask()
