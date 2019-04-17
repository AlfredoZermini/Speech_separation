import numpy as np
import os
import traceback
import soundfile as sf
from random import shuffle
from scipy import signal
import itertools
import random
import sys
import h5py

def generate_list(task):
    
    threshold = 35000
   
    project_path = '/vol/vssp/mightywings'
    timit_path = os.path.join(project_path, 'TIMIT', str(task))
    print timit_path

    # read the whole timit database and create a complete list of speech files
    dr_path = timit_path
    dirs = [d for d in os.listdir(dr_path) if os.path.isdir(os.path.join(dr_path, d))]
    files_list = []
    prev_male_list = []
    prev_female_list = []
   
    for dr in dirs:
      
        subdirs = [d for d in os.listdir(os.path.join(dr_path,dr)) if os.path.isdir(os.path.join(os.path.join(dr_path,dr), d))]
        
        for subdr in subdirs:
           
            files = os.listdir(os.path.join(dr_path,dr,subdr))
             
            for file in files:

               if file.endswith(".wav"):
                  files_list.append(os.path.join(dr_path,dr,subdr,file))

    # generate list without invalid files
    good_files_list = []

    for path in files_list:
        
        try:
            data = sf.read(path)
            a = len(data[0])
            if a > threshold:
                good_files_list.append(path)
        
        except Exception:
           #traceback.print_exc()
           continue
           

    # evaluate minimum length
    min_length = 100000
    path_pos = -1
    min_path_pos = 0
    
    for path in good_files_list:
        
        data = sf.read(path)
        a = len(data[0])
        
        if a < min_length:
            min_length = a
            min_path_pos = path_pos

    
    return good_files_list, min_length