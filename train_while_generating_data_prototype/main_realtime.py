#CUDA_VISIBLE_DEVICES=1 python main_mask.py B_format train 1 ['theta','MV'] '' '3targets_3layers_halfsize_200epochs' 
import sys
import numpy as np
import os
import time
import atexit
import h5py
from train_RNN_mask import train #train_spatial_keras
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# hyper params
n_sources = 3
n_hid = 256/2
n_epochs = 300 #60 170
batch_size = 1 #6 DNN and CNN, 2 RNN, 2 big CNN and DNN
n_masks = 3

def main_mask():

    Xtrain, y = prepare_input_mask(vel, p0)
        
    # labels
    ytrain0 = y[:,0,:,:]
    ytrain1 = y[:,1,:,:]
    ytrain2 = y[:,2,:,:]
         
    # train
    model = train(Xtrain, ytrain0, ytrain1, ytrain2, n_hid, n_epochs, batch_size, n_masks, sys.argv[4])
    
    # set DNNs save path
    model.save(DNN_name)
    
    print "%s - Saving models into:\n%s\n" % (time.ctime(), DNN_name)
    print "%s - Done!" % time.ctime()


