from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from tensorflow.keras import losses
from tensorflow.keras import backend as K
import random
import math
from tqdm import tqdm

def create_orderN(y_noise, order):
    if order==1:
        kernel = np.array([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
    if order==2:
        kernel = np.array([
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]])
    kernel = kernel[..., tf.newaxis, tf.newaxis]
    kernel = tf.constant(kernel, dtype=np.float32)
    y_order = tf.nn.conv2d(y_noise, kernel, strides=[1, 1, 1, 1], padding='SAME')
    y_order = tf.clip_by_value(y_order, clip_value_min=0., clip_value_max=1.)
    y_order -= y_noise
    y_order = tf.clip_by_value(y_order, clip_value_min=0., clip_value_max=1.)
    return y_order

# only hits -> 1
def get_hit_average():
    @tf.autograph.experimental.do_not_convert
    def hit_average(data, y_pred):
        y_true = data[:,:,:,0]
        nofHits = tf.math.count_nonzero(tf.greater(y_true,0.01), dtype=tf.float32)
        return (K.sum(y_true*y_pred[:,:,:,0])/nofHits)
    return hit_average

# hits in range 1 (like kernel 3x3) around noise pixel -> 1
def get_hit_average_order1():
    @tf.autograph.experimental.do_not_convert
    def hit_average_order1(data, y_pred):
        y_hits_in_order1 = data[:,:,:,0]*data[:,:,:,2]
        nofHitsInOrder1 = tf.math.count_nonzero(tf.greater(y_hits_in_order1,0.01), dtype=tf.float32)
        return (K.sum(y_hits_in_order1*y_pred[:,:,:,0])/nofHitsInOrder1)
    return hit_average_order1

# hits in range 2 (like kernel 5x5) around noise pixel -> 1
def get_hit_average_order2():
    @tf.autograph.experimental.do_not_convert
    def hit_average_order2(data, y_pred):
        y_hits_in_order2 = data[:,:,:,0]*data[:,:,:,3]
        nofHitsInOrder2 = tf.math.count_nonzero(tf.greater(y_hits_in_order2,0.01), dtype=tf.float32)
        return (K.sum(y_hits_in_order2*y_pred[:,:,:,0])/nofHitsInOrder2)
    return hit_average_order2       

# only noise -> 0
def get_noise_average():
    @tf.autograph.experimental.do_not_convert
    def noise_average(data, y_pred):
        y_noise = data[:,:,:,1]
        nofNoise = tf.math.count_nonzero(tf.greater(y_noise,0.01), dtype=tf.float32)
        return (K.sum(y_noise*y_pred[:,:,:,0])/nofNoise)
    return noise_average

# empty pmt (no hits/noise pixels!) -> 0
def get_background_average():  
    @tf.autograph.experimental.do_not_convert
    def background_average(data, y_pred):
        y_true = data[:,:,:,0]
        y_noise = data[:,:,:,1]
        y_background = tf.clip_by_value(-y_true - y_noise + tf.constant(1.0), clip_value_min=0., clip_value_max=1.)
        nofBackground = tf.math.count_nonzero(y_background, dtype=tf.float32)
        return (K.sum(K.abs(y_background*y_pred[:,:,:,0]))/nofBackground)
    return background_average 

# custom loss function to be able to use noise_train/test in loss/metrics
# data[:,:,:,0] = hits_train/hits_test , data[:,:,:,1] = noise_train/noise_test
def get_custom_loss():
    @tf.autograph.experimental.do_not_convert
    def custom_loss(data, y_pred):
        y_true = data[:,:,:,0]
        return losses.mean_squared_error(y_true, y_pred[:,:,:,0]) 
    return custom_loss

# load data file and some preprocessing 
def loadDataFile(datafile, nofRows=20000, pixel_x = 32, pixel_y = 72):
    with open(datafile, 'r') as temp_f:
        col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
    column_names = [i for i in range(0, max(col_count))]
    hits = pd.read_csv(datafile,header=None ,index_col=0,comment='#', delimiter=",", nrows= nofRows,names=column_names).values.astype('int32')
    hits[hits < 0] = 0
    hits_temp = np.zeros([len(hits[:,0]), pixel_x*pixel_y])
    for i in range(len(hits[:,0])):
        for j in range(len(hits[0,:])):
            if hits[i,j]==0:
                break
            hits_temp[i,hits[i,j]-1]+=1
    hits_temp = tf.reshape(hits_temp, [len(hits[:,0]), pixel_y, pixel_x])
    hits_temp = tf.clip_by_value(hits_temp, clip_value_min=0., clip_value_max=1.)
    hits = tf.cast(hits_temp[..., tf.newaxis],dtype=tf.float32)
    print('load data from  ' + datafile + '  -> ' + str(len(hits[:])) + '  events loaded' )
    return hits

def random_p(p):
    if(random.betavariate(5,2) < p):
        return 1
    else:
        return 0

def getPixelNr(pixelId):
    y = (pixelId-1)//32
    x = ((pixelId-1)%32)
    return x, y


def createNoiseFromFile(datafile, p=0.7, pixel_x=32, pixel_y=72):
    with open(datafile, 'r') as temp_f:
        col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
    column_names = [i for i in range(0, max(col_count))]
    hits = pd.read_csv(datafile,header=None ,index_col=0,comment='#', delimiter=",", nrows= 20000,names=column_names).values.astype('int32')
    hits[hits < 0] = 0
    
    hits_temp = np.zeros([len(hits[:,0]), pixel_y, pixel_x])
    nofNoiseHits = 0
    for i in tqdm(range(len(hits[:,0]))):
        for j in range(len(hits[0,:])):
            if hits[i,j]==0:
                break
            nofNoiseHits += 1
            x, y = getPixelNr(hits[i,j])
            for k in range(-2, 2, 1):
                for l in range(-2, 2, 1):
                    if ((x+k)<0 or (x+k)>31) or ((y+l)<0 or (y+l)>71) or (k==0 and l==0):
                        continue
                    elif (nofNoiseHits % 10 == 0):
                        hits_temp[i, y+l, x+k]+= random_p(2.0*p/math.sqrt(k*k+l*l))
                    elif (nofNoiseHits % 3 == 0):
                        hits_temp[i, y+l, x+k]+= random_p(1.6*p/math.sqrt(k*k+l*l))
                    else :
                        hits_temp[i, y+l, x+k]+= random_p(1.15*p/math.sqrt(k*k+l*l))
            hits_temp[i, y, x]=1
    hits_temp = tf.clip_by_value(hits_temp, clip_value_min=0., clip_value_max=1.)
    hits = tf.cast(hits_temp[..., tf.newaxis],dtype=tf.float32)
    print('loaded data from  ' + datafile + '  ->   noise generated!' )        
    return hits

def get_hits_concat(hits_all, hits_true):
    noise = tf.clip_by_value(tf.math.add(hits_all, -hits_true),clip_value_min=0., clip_value_max=1.)
    order1 = create_orderN(noise, 1)
    order2 = create_orderN(noise, 2)
    order2 -= order1
    order2 = tf.clip_by_value(order2, clip_value_min=0., clip_value_max=1.)
    return hits_all, tf.concat([hits_true, noise, order1, order2], 3)



def single_event_plot(data, data0, nof_pixel_X, min_X, max_X, nof_pixel_Y, min_Y, max_Y, eventNo, label_pred, cut=0.):
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(tf.cast(data[eventNo] > cut, data[eventNo].dtype) * data[eventNo], interpolation='none', extent=[min_X,max_X,min_Y,max_Y], cmap='gray')
    plt.title('sim = {:.3f}   real = {:.3f}'.format(label_pred[eventNo,0], label_pred[eventNo,1]))
    #y = tf.maximum(data[eventNo], 0.5)
    plt.colorbar()
    #plt.gray()
    ax = plt.subplot(1, 2, 2)
    cmap = colors.ListedColormap(['black','white', 'red', 'grey'])
    bounds = [0,0.1,1.25,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(data0[eventNo], interpolation='none', extent=[min_X,max_X,min_Y,max_Y], cmap=cmap, norm=norm)
    plt.title("original")
    #plt.colorbar()
    plt.show()
    return
