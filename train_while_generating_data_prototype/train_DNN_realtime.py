import sys
import numpy as np
import os
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Convolution2D, LeakyReLU, Reshape, MaxPooling2D, InputLayer, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import plot_model, to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

# neighbour parameters
frame_neigh = 5
n_frames = frame_neigh*2+1   

figures_path = os.path.join('/user/cvssppgr/az00147/Documents/Python/B_format/Figures/masks_test')

# custom cost function
def cost(y_true, y_pred):
   
   #return K.mean(K.sigmoid())  (K.square(y_true-y_pred)) )
   return K.mean( K.square(y_true-y_pred ) ) 
   
def cost1(y_true, y_pred):

    IBM = y_true[:,0:127]
    mixtureLP = y_true[:,127:254]
    groundthLP = y_true[:, 254:381]

    # estimateLP = (log(mx)^2 - u)/sigma = 2log(m)/sigma + mixtureLP
    estimateLP = (2 / sig_std) * K.log(IBM) + mixtureLP

    diff = K.square(y_pred - IBM)

    weight1 = K.sigmoid(groundthLP)
    weight2 = K.sigmoid(estimateLP)

    weight = (1-weight1)*weight2 + weight1

    return K.sum(diff*weight, axis=-1)   

   
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

   
def generator(Xtrain, ytrain0, ytrain1, ytrain2, batch_size, n_masks):
   
   [n_samples,n_freq,n_time,n_features] = Xtrain.shape
   
   # batch counter
   idx = 0
   
   # Create empty arrays to contain batch of features and labels#
   batch_features = np.zeros([batch_size, n_freq, n_time, n_features])
   batch_masks = np.zeros([batch_size, n_freq, n_time, n_masks])
 
   while True:
      #print range(batch_size)
      for i in range(batch_size):
         
         # choose random index in features
         index = np.random.choice(Xtrain.shape[0],1)
         print index
         
         # pick random batches
         batch_features[i] = Xtrain[index,:,:,:]
         batch_masks[i,:,:,0] = ytrain0[index]
         
         if n_masks > 1:
            batch_masks[i,:,:,1] = ytrain1[index]
            batch_masks[i,:,:,2] = ytrain2[index]
      
         # create neighbour frame information
         X_in = neighbour1(batch_features,frame_neigh)
         
         ### labels
         # create neighbour frame information
         Y_out = neighbour1(batch_masks,frame_neigh)
         
         # pick only masks central frames in order to predict just the real frame, not its neighbours
         Y_out = Y_out[:,:,frame_neigh,:]
         
         
         if n_masks == 1:
            yield [X_in], [Y_out[:,:,0]]
         else: 
            yield [X_in], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2]]
         
         print('number of batches = %d' % idx)
         idx += 1


### train
def train(Xtrain, ytrain0, ytrain1, ytrain2, n_hid, n_epochs, batch_size, n_masks, features):    

   ### input data
   # load data
   [n_samples, n_freq, n_time, n_features] = Xtrain.shape
   
   input_img = Input(shape=(n_freq, n_frames, n_features ) )
   x = ( Flatten())(input_img)
   
   if n_masks == 1:
       
      for i in range(0,4):
        x = ( Dense(1024))(x)
        x = ( BatchNormalization() )(x)
        x = ( LeakyReLU())(x)

      o0 = ( Dense(n_freq, activation='sigmoid'))(x)

      DNN = Model(input_img, [o0])
      DNN.summary()
      
   else:

      for i in range(0,4):
        x = ( Dense(1024))(x)
        x = ( BatchNormalization() )(x)
        x = ( LeakyReLU())(x)
      o0 = ( Dense(n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(n_freq, activation='sigmoid'))(x)
      o2 = ( Dense(n_freq, activation='sigmoid'))(x)

      DNN = Model(input_img, [o0,o1,o2])
      DNN.summary()

   # compile
   adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.95, decay=0.0)
   sgd = SGD(lr=0.1, momentum=0.9, decay=0, nesterov=False)
   DNN.compile(optimizer=sgd, loss=cost, metrics=[cost])
   
   # fit model
   #DNN.fit(Xnew, [Ynew[:,:,0],Ynew[:,:,1],Ynew[:,:,2]], epochs=n_epochs, batch_size=batch_size)
   DNN.fit_generator(generator(Xtrain, ytrain0, ytrain1, ytrain2, batch_size, n_masks), samples_per_epoch=n_samples/batch_size, nb_epoch=n_epochs)
    
   print "\n%s - Finished training" % (time.ctime())
   
   return DNN
    
    
if __name__ == '__main__':
   train_spatial()
   K.clear_session()
