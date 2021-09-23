# %%
import random
import math
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def random_p(p):
    if(random.betavariate(5,2) < p):
        return 1
    else:
        return 0

def sim_centerhits(p):
    sample_size = 10000
    temp = np.zeros([sample_size, 5, 5])
    for i in tqdm(range(sample_size)):
        for k in range(-2, 2, 1):
            for l in range(-2, 2, 1):
                if (k==l==0):
                    continue 
                elif (i % 10 == 0):
                    temp[i, l+2, k+2]+= random_p(2.0*p/math.sqrt( k*k+l*l ))
                elif (i % 3 == 0):
                    temp[i, l+2, k+2]+= random_p(1.6*p/math.sqrt( k*k+l*l ))
                else :
                    temp[i, l+2, k+2]+= random_p(1.15*p/math.sqrt( k*k+l*l ))
        temp[i, 2, 2]=1
    return temp

def count_hits(array):
    count = np.zeros(26)
    for i in tqdm(range(len(array[:,0,0]))):
        value = tf.math.count_nonzero(array[i,:,:], dtype=tf.int32)
        count[value.numpy()] += 1
    count[0] = count[2]
    return count
# %%
x = np.arange(26)
y = count_hits(sim_centerhits(0.475))

n = sum(y)
mean = sum(x*y)/n
sigma = sum(y*(x-mean)**2)/n
print(mean)
print(sigma)
plt.bar(x, y)
#plt.xticks(y_pos, bars)
plt.show()


# %%
