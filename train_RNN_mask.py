import sys
import numpy as np
import os
import time
from keras import losses
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Convolution2D, LeakyReLU, Reshape, MaxPooling2D, InputLayer, BatchNormalization, Dropout, Activation, GRU, LSTM, Bidirectional
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model, to_categorical
from keras import backend as K
import matplotlib.pyplot as plt

figures_path = os.path.join('/user/cvssppgr/az00147/Documents/Python/B_format/Figures/masks_test')

frame_neigh = 1
n_frames = frame_neigh*2+1  

# custom cost function
def cost(y_true, y_pred):
   
   #return K.mean(K.sigmoid())  (K.square(y_true-y_pred)) )
   return K.mean( K.square(y_true-y_pred ) ) 
   
# creates spectrograms with neighbour frames

def neighbour1(X, frame_neigh):
   
   [n_samples1,n_freq,n_time,n_features] = X.shape 

   # define tensors
   X_neigh = np.zeros([n_samples1*n_time, n_freq, n_frames, n_features ])
   X_zeros = np.zeros([n_samples1, n_freq, n_time+frame_neigh*2, n_features ])
   
   # create X_zeros, who is basically X with an empty border of 2*frame_neigh frames
   X_zeros[:,:,frame_neigh:X_zeros.shape[2]-frame_neigh,:] = X
   
   for sample in range(0,n_samples1 ):
      for frame in range(0,n_time ):
         X_neigh[sample*n_time+frame, :, :, :] = X_zeros[sample, :, frame:frame+n_frames, :]
      
   return X_neigh

def generator_group(Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, batch_size, n_sources):
   
   [n_samples,n_freq,n_time,n_features] = Xtrain.shape
   
   # batch counter
   idx = 0
   
   # Create empty arrays to contain batch of features and labels#
   batch_features = np.zeros([batch_size, n_freq, n_samples*n_time, n_features])
   batch_masks = np.zeros([batch_size, n_freq, n_samples*n_time, n_sources])
 

   Xtrain_group = np.zeros([1,n_freq,n_samples*n_time,n_features])
   ytrain0_group = np.zeros([1,n_freq,n_samples*n_time])
   ytrain1_group = np.zeros([1,n_freq,n_samples*n_time])
   ytrain2_group = np.zeros([1,n_freq,n_samples*n_time])

   for i in range(0, n_samples):
      Xtrain_group[:,:, i*n_time:i*n_time+n_time ,:] = Xtrain[i,:,:,:]
      ytrain0_group[:,:, i*n_time:i*n_time+n_time] = ytrain0[i,:,:]
      ytrain1_group[:,:, i*n_time:i*n_time+n_time] = ytrain1[i,:,:]
      ytrain2_group[:,:, i*n_time:i*n_time+n_time] = ytrain2[i,:,:]
      
   while True:
      #print range(batch_size)
      for i in range(batch_size):
         
         # choose random index in features
         index = np.random.choice(Xtrain_group.shape[0],1)
         print index
         
         # pick random batches
         batch_features[i] = Xtrain_group[index,:,:,:]
         batch_masks[i,:,:,0] = ytrain0_group[index]
         
         if n_sources > 1:
            batch_masks[i,:,:,1] = ytrain1_group[index]
            batch_masks[i,:,:,2] = ytrain2_group[index]
      
         # create neighbour frame information
         X_in = neighbour1(batch_features,frame_neigh)
         
         Xnew = np.zeros([X_in.shape[0], X_in.shape[1]*X_in.shape[3], X_in.shape[2] ] )
         
         for feat in range(0,X_in.shape[3]):
               Xnew[:,X_in.shape[1]*feat:X_in.shape[1]*feat+X_in.shape[1],:] = X_in[:,:,:,feat]
         

         
         ### labels
         # create neighbour frame information
         Y_out = neighbour1(batch_masks,frame_neigh)
         
         # pick only masks central frames in order to predict just the real frame, not its neighbours
         Y_out = Y_out[:,:,frame_neigh,:]
         
         
         if n_sources == 1:
            yield [Xnew], [Y_out[:,:,0]]
         else: 
            print Xnew.shape
            print Y_out.shape
            cocco
            yield [Xnew], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2]]
         
         print('number of batches = %d' % idx)
         idx += 1
         
def generator(Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, batch_size, n_sources):
   
   [n_samples,n_freq,n_time,n_features] = Xtrain.shape
   
   # batch counter
   idx = 0
   
   # Create empty arrays to contain batch of features and labels#
   batch_features = np.zeros([batch_size, n_freq, n_time, n_features])
   batch_masks = np.zeros([batch_size, n_freq, n_time, n_sources])
 
   while True:
      #print range(batch_size)
      for i in range(batch_size):
         
         # choose random index in features
         index = np.random.choice(Xtrain.shape[0],1)
         print index
         
         # pick random batches
         batch_features[i] = Xtrain[index,:,:,:]
         batch_masks[i,:,:,0] = ytrain0[index]
         
         if n_sources > 1:
            batch_masks[i,:,:,1] = ytrain1[index]
            batch_masks[i,:,:,2] = ytrain2[index]
      
         # create neighbour frame information
         X_in = neighbour1(batch_features,frame_neigh)
         
         Xnew = np.zeros([X_in.shape[0], X_in.shape[1]*X_in.shape[3], X_in.shape[2] ] )
         
         for feat in range(0,X_in.shape[3]):
               Xnew[:,X_in.shape[1]*feat:X_in.shape[1]*feat+X_in.shape[1],:] = X_in[:,:,:,feat]

         
         ### labels
         # create neighbour frame information
         Y_out = neighbour1(batch_masks,frame_neigh)
         
         # pick only masks central frames in order to predict just the real frame, not its neighbours
         Y_out = Y_out[:,:,frame_neigh,:]
         
         
         if n_sources == 1:
            yield [Xnew], [Y_out[:,:,0]]
         else: 
            print Xnew.shape
            print Y_out
            cocco
            yield [Xnew], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2]]
         
         print('number of batches = %d' % idx)
         idx += 1



def train(RNN_path, Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, n_hid, n_epochs, batch_size, n_sources, features):
    

### train

   ### input data
   # load data
   [n_samples, n_freq, n_time, n_features] = Xtrain.shape

   input_img = Input(shape=(n_freq*n_features,n_frames ) )
   x = Bidirectional(LSTM(n_hid,return_sequences=True, dropout=0.5, recurrent_dropout=0.2))(input_img)
   #x = Dropout(0.2)(x)
   x = Bidirectional(LSTM(n_hid,  dropout=0.5, recurrent_dropout=0.2))(x)
   #x = Dropout(0.2)(x)
   o0 = ( Dense(n_freq, activation='sigmoid'))(x)
   o1 = ( Dense(n_freq, activation='sigmoid'))(x)
   o2 = ( Dense(n_freq, activation='sigmoid'))(x)

   if n_sources == 3:
      RNN = Model(input_img, [o0,o1,o2])
   elif n_sources == 4:
      RNN = Model(input_img, [o0,o1,o2,o3])
   
   RNN.summary()

   # compile
   adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.95, decay=0.0)
   sgd = SGD(lr=0.5, momentum=0.9, decay=1e-6, nesterov=False)
   RNN.compile(optimizer=sgd, loss=cost, metrics=[cost])
   
   # save after every epoch
   checkpoint = ModelCheckpoint(RNN_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
   
   # fit model
   #DNN.fit(Xnew, [Ynew[:,:,0],Ynew[:,:,1],Ynew[:,:,2]], epochs=n_epochs, batch_size=batch_size)
   RNN.fit_generator(generator_group(Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, batch_size, n_sources), samples_per_epoch=n_samples/batch_size, callbacks=[checkpoint], nb_epoch=n_epochs)
    
   print "\n%s - Finished training" % (time.ctime())
   
   return RNN
    
    
if __name__ == '__main__':
   train_spatial()
   K.clear_session()
