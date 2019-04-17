# Speech_separation
Python code used in my PhD

1)List of .py files in alphabetical order:

-feature.py -> evaluate low level features

-generate_list.py -> generate training and testing files lists

-generate_spectrograms_mask.py -> generate the spectrograms masks

-main_mask.py -> run neural network training

-prepare_data.py -> create speech mixtures

-prepare_input_mask.py -> create input features and stores them

-stats -> running stats

-train_CNN_mask.py -> train CNN

-train_DNN_mask.py -> train DNN

-train_RNN_mask.py -> train RNN

-utilities.py -> more utilities




2)Code hierarchy (only the main parts of the code are listed):

    prepare_input_mask.py

        |___prepare_data.py - stats.py
 
            |___generate_list.py
      
 
    main_mask.py

        |___train_DNN_mask.py - train_CNN_mask.py - train_RNN_mask.py




3)Instructions

Modify the project path '/vol/vssp/mightywings/'.

The Code hierarchy section contains the three main pieces of codes that the user needs to execute: prepare_input_mask.py, main_mask.py and run_test_mask.py. 

Their very first line of each of these .py files corresponds to the command the user needs to run, for example:

python prepare_input_mask.py B_format train 12BB01 12BB01 ['theta','MV'] ''

will execute prepare_input_mask.py, with several given options, depending on the task. 

In this case, the options are: 'B_format' is the name of the folder, 'train' or 'test' are the two possible tasks, '12BB01' can be either the training or the testing room and ['theta','MV'] are the low-level features used.


Run the code in the following order:

a)Run prepare_input_mask.py to generate the training and testing sets. Set the correct parameters first for the training and then the testing features.

b)Run main_mask.py to run the training of the neural network. A DNN, CNN or RNN can be selected by importing the correct .py. i.e. import train_DNN_mask.py will run a DNN.
