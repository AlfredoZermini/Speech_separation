import inspect
import wavio
import matplotlib.pyplot as plt
import numpy as np

# get variable name as string
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

# plot figure    
def plot_figure(var, path_to_figure, x_label, y_label):

    var_name = retrieve_name(var) 

    plt.imshow(var[:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
    clb = plt.colorbar()
    clb.ax.set_title(var_name)
    plt.xlabel(retrieve_name(x_label))
    plt.ylabel(retrieve_name(y_label))
    plt.title(var_name)
    plt.gca().invert_yaxis()
    plt.savefig(path_to_figure)
    plt.clf()
    
# get frame
def gettimeframes(vec):
    global n_bin
    n_bin = len(vec[0])
    return n_bin
    
# readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs
    
