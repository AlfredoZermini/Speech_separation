#CUDA_VISIBLE_DEVICES=1 python main_mask.py B_format train 12BB01 ['theta','MV'] '' '' 
import sys
import numpy as np
import os
import time
import atexit
import h5py
from train_DNN_mask import train #train_spatial_keras
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt

# hyper params
n_sources = 3
n_hid = 256/2 #256/2
n_epochs = 300 #60 170
batch_size = 6 #6 DNN and CNN, 2 RNN, 2 big CNN and DNN

def main_mask():

    # dataset
    if 'theta' in features_name and 'MV' not in features_name:
        Xtrain = x[:,:,:,0]
        Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2],1)
                
    elif 'MV' in features_name and 'theta' not in features_name:
        Xtrain = np.zeros([x.shape[0],x.shape[1],x.shape[2],2])
        Xtrain = x[:,:,:,1:3]
        #Xtrain = x[:,:,:,1:3]
        
    elif 'theta' in features_name and 'MV' in features_name:
        Xtrain = np.zeros([x.shape[0],x.shape[1],x.shape[2],2])
        #Xtrain[:,:,:,0] = x[:,:,:,0]
        #Xtrain[:,:,:,1] = np.mean(x[:,:,:,1:3],axis=3)
        Xtrain = x

    elif 'theta' in features_name and 'MV' in features_name and 'MVratio' in features_name:
        print x.shape
        Xtrain = np.zeros([x.shape[0],x.shape[1],x.shape[2],3])
        
        Xtrain[:,:,:,0] = x[:,:,:,0]
        Xtrain[:,:,:,1] = np.mean(x[:,:,:,1:3],axis=3)
        Xtrain[:,:,:,2] = x[:,:,:,1]/x[:,:,:,2]
        print Xtrain.shape
        print x.shape
        
    # labels
    print y.shape
    ytrain0 = y[:,0,:,:]
    ytrain1 = y[:,1,:,:]
    ytrain2 = y[:,2,:,:]
    ytrain3 = 0
    
    if n_sources == 4:
        ytrain3 = y[:,3,:,:]
        
    # train
    model = train(DNN_path,Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, n_hid, n_epochs, batch_size, n_sources, sys.argv[4])
    
    print "%s - Done!" % time.ctime()

if __name__ == '__main__':

    
    if sys.argv[1] != "" and sys.argv[2].lower() != '' and sys.argv[3] in ['1','12BB01'] and set(['MV','theta','MVratio']).issuperset(set(sys.argv[4].replace("[","").replace("]","").split(","))):
            
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
        
        # load path
        load_TrainData_path = os.path.join(project_path, 'InputData/TrainData/Room' + Train_Room + '/'  + task + '_TrainRoom' + Train_Room  + '_' + n_sources_string + 'Sources' + post_suffix + '.h5')
        print load_TrainData_path

        # load input data
        print "%s - Loading input data from:\n%s\n" % (time.ctime(), load_TrainData_path)
        with h5py.File(load_TrainData_path, 'r') as hf:
            print hf.keys()
            
            x = hf.get('data')
            y = hf.get('IRM') #IRM    
            
            x = np.array(x)
            y = np.array(y)   
               
        print x.shape
        
        # DNN folder
        results_path = os.path.join(project_path, 'Results/Room' + Train_Room )
        DNN_path = os.path.join(results_path, 'DNN_models_' + suffix + '_' + task + post_suffix + DNN_suffix + '_' + n_sources_string + '.h5')
        
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        main_mask()

    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()
