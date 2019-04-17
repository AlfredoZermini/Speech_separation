import sys
import numpy as np
import os
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Convolution2D, LeakyReLU, Reshape, MaxPooling2D, InputLayer, BatchNormalization, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils import plot_model, to_categorical
from keras import backend as K
import matplotlib.pyplot as plt

# neighbour parameters
frame_neigh = 5
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
         
         # pick random batches
         batch_features[i] = Xtrain[index,:,:,:]
         batch_masks[i,:,:,0] = ytrain0[index]
         batch_masks[i,:,:,1] = ytrain1[index]
         batch_masks[i,:,:,2] = ytrain2[index]
         
         if n_sources == 4: 
            batch_masks[i,:,:,3] = ytrain3[index]
      
      # create neighbour frame information
      X_in = neighbour1(batch_features,frame_neigh)
         
      ### labels
      # create neighbour frame information
      Y_out = neighbour1(batch_masks,frame_neigh)
         
      # pick only masks central frames in order to predict just the real frame, not its neighbours
      Y_out = Y_out[:,:,frame_neigh,:]

      if n_sources == 3:
         yield [X_in], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2]]
      elif n_sources == 4: 
         yield [X_in], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2], Y_out[:,:,3]]
         
      print('number of batches = %d' % idx)
      idx += 1


def train(DNN_path,Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, n_hid, n_epochs, batch_size, n_sources, features):
    

### train

   ### input data
   # load data
   [n_samples, n_freq, n_time, n_features] = Xtrain.shape

   input_img = Input(shape=(n_freq,n_frames,n_features ) )

   x = ( Convolution2D(64, kernel_size=(5, 5), activation='linear', padding='same' ))(input_img)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   x = ( MaxPooling2D(pool_size=(2, 2)))(x)
   x = ( Convolution2D(128, kernel_size=(3, 3), activation='linear', padding='same' ) )(x)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   x = ( MaxPooling2D(pool_size=(2, 1)))(x)
   x = ( Convolution2D(256, kernel_size=(8, 1), activation='linear', padding='same' ) )(x)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   x = ( MaxPooling2D(pool_size=(4, 1)))(x)
   x = ( Flatten())(x)
   x = ( Dense(1024*n_sources, activation='relu'))(x)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   o0 = ( Dense(n_freq, activation='sigmoid'))(x)
   o1 = ( Dense(n_freq, activation='sigmoid'))(x)
   o2 = ( Dense(n_freq, activation='sigmoid'))(x)

   CNN = Model(input_img, [o0,o1,o2])
   CNN.summary()
   
   # compile
   #adam = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=0.95, decay=0.0)
   sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
   CNN.compile(optimizer=sgd, loss=cost, metrics=[cost])
   
   # save after every epoch
   checkpoint = ModelCheckpoint(DNN_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
   
   # fit model
   #CNN.fit(Xnew, [Ynew[:,:,0],Ynew[:,:,1],Ynew[:,:,2]], epochs=n_epochs, batch_size=batch_size)
   CNN.fit_generator(generator(Xtrain, ytrain0, ytrain1, ytrain2, ytrain3, batch_size, n_sources), samples_per_epoch=n_samples/batch_size, callbacks = [checkpoint], nb_epoch=n_epochs)
    
   print "\n%s - Finished training" % (time.ctime())
   
   return CNN
    
    
if __name__ == '__main__':
   train_spatial()
   K.clear_session()
