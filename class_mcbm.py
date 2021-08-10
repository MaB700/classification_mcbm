# %%
from __future__ import absolute_import, division, print_function, unicode_literals
from math import sqrt
from numpy.lib.npyio import load
import os
import pandas as pd
import numpy as np
import seaborn as sn
import ipywidgets as widgets
from ipywidgets import fixed
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import math
#os.environ['AUTOGRAPH_VERBOSITY'] = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            #tf.config.experimental.set_virtual_device_configuration(gpus[0], \
                #[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, BatchNormalization, AveragePooling2D
from tensorflow.keras.models import Sequential

# load custom functions/loss/metrics
from class_mcbm_functions import *

import wandb
from wandb.keras import WandbCallback
#wandb.init(project="autoencoder_mcbm_toy_denoise")

print('Tensorflow version: ' + tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")
# %%
def random_p(p):
    if(random.random() < p):
        return 1
    else:
        return 0

def random_kernel(p):
    kernel = np.array([
                        [0, 0, 0, 0, 0],
                        [0, random_p(p/sqrt(2)), random_p(p), random_p(p/sqrt(2)), 0],
                        [0, random_p(p), 1, random_p(p), 0],
                        [0, random_p(p/sqrt(2)), random_p(p), random_p(p/sqrt(2)), 0],
                        [0, 0, 0, 0, 0]])
    #print(kernel)
    return kernel

# %%
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
""" hits_all_train, hits_train = get_hits_concat(loadDataFile("E:/ML_data/mcbm_rich/28.07/hits_all.txt"), \
                                             loadDataFile("E:/ML_data/mcbm_rich/28.07/hits_true.txt"))
hits_all_test, hits_test = get_hits_concat(loadDataFile("E:/ML_data/mcbm_rich/28.07/hits_all_test.txt"), \
                                           loadDataFile("E:/ML_data/mcbm_rich/28.07/hits_true_test.txt")) """

def load_data_class(hits_sim, hits_real):
    label_sim = tf.constant(0, shape=(len(hits_sim[:])))
    label_real = tf.constant(1, shape=(len(hits_real[:])))
    hits = tf.concat([hits_sim, hits_real], 0)
    labels = tf.concat([label_sim, label_real], 0)
    hits_trainx, hits_testx, label_trainx, label_testx = train_test_split(hits.numpy(), labels.numpy(), test_size=0.25)
    # maybe transform to tf tensor tf.convert_to_tensor(.....)
    return tf.convert_to_tensor(hits_trainx), tf.convert_to_tensor(hits_testx), \
           tf.convert_to_tensor(label_trainx), tf.convert_to_tensor(label_testx)

hits_sim = loadDataFile("E:/ML_data/mcbm_rich/28.07/hits_all.txt")
hits_real = loadDataFile("E:/ML_data/mcbm_rich/real/hits_real.txt")
hits1, hits2, label1, label2 = load_data_class(hits_sim, hits_real) #(hits_train[:,:,:,0])[..., tf.newaxis]


custom_metrics = [get_hit_average(), get_noise_average(), get_background_average(),\
                    get_hit_average_order1(), get_hit_average_order2()]
# %%
#print(label1. add axis
# %%
load_weights = 0
cp_path = "tmp/cp.ckpt"
cp_dir = os.path.dirname(cp_path)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=cp_path,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

model = Sequential()
model.add(Input(shape=(72, 32, 1)))

model.add(Conv2D(filters=16, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2))) 


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(Dense(1, activation='sigmoid', use_bias=False))
#model.add(tf.keras.layers.Softmax())

""" model.add(Conv2DTranspose(filters=128 , kernel_size=3, strides=[2, 2],activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=64 , kernel_size=3, strides=[2, 2],activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=32 , kernel_size=3, strides=2,activation='relu', padding='same'))
model.add(Conv2D(1, kernel_size=3, activation='tanh', padding='same')) """
model.summary()

if (load_weights == 1):
    print('loading weights ...\n')
    model.load_weights(cp_path)
    print('weights loaded! \n')

#opt = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-07 )
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], experimental_steps_per_execution=10)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
model.fit(hits1, label1,
                epochs=50,
                batch_size=500,
                shuffle=True,
                validation_data=(hits2, label2),
                callbacks=[es])#,
                #callbacks=[WandbCallback(log_weights=True)])
#print('model evaluate ...\n')
# %%
label_sim_pred = model.predict(hits_sim)
print(tf.reduce_mean(label_sim_pred))
label_real_pred = model.predict(hits_real)
print(tf.reduce_mean(label_real_pred))
# %%
label_pred = model.predict(hits2)
fpr, tpr, th = roc_curve(label2, label_pred)
auc = auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='auc = {:.3f}'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# # single_event_plot(hits_noise_train, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)
# %%
np.set_printoptions(precision=2)
cm = confusion_matrix(label2, np.around(label_pred).astype(int), normalize='true')
cm = np.around(cm,2)
df_cm = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])
plt.figure(figsize=(10,7))
sn.heatmap(df_cm, annot=True)
plt.xlabel('predicted labels')
plt.ylabel('true labels')
plt.show()
# %%
original_plt = tf.math.add(hits_test[:,:,:,0], tf.math.scalar_mul(2.0, hits_test[:,:,:,1]) )
encoded = model.predict(hits_all_test, batch_size=50)
# %%
print(get_hit_average()(hits_test, tf.keras.layers.ReLU(threshold=0.7)(encoded)).numpy())

def f1(thresholdx, mode, pred=encoded):
    result = np.array([-1.])
    for th in thresholdx:
        if (mode==0):
            av = get_hit_average()(hits_test, tf.keras.layers.ReLU(threshold=th)(pred)) 
        elif (mode==1):
            av = 1-get_noise_average()(hits_test, tf.keras.layers.ReLU(threshold=th)(pred))
        result = np.append(result, av)
    result = np.delete(result, 0)
    return result

t1 = np.arange(0., 1., 0.01)

fig, ax = plt.subplots()
ax.plot(t1, f1(t1, mode=0), '-', label='hit average')
ax.plot(t1, f1(t1, mode=1), '-', label='1-noise average')
ax.set(xlabel='cut', ylabel='average value')
ax.legend()
plt.show()


# %%
interactive_plot = widgets.interact(single_event_plot, \
                    data=fixed(tf.squeeze(hits_real,[3])), data0=fixed(tf.squeeze(hits_sim,[3])), \
                    nof_pixel_X=fixed(32), min_X=fixed(-8.1), max_X=fixed(13.1), \
                    nof_pixel_Y=fixed(72), min_Y=fixed(-23.85), max_Y=fixed(23.85), eventNo=(50,100-1,1), cut=(0.,0.90,0.05))



# # %%
# #model.evaluate((hits_noise_test[14,:,:,:])[tf.newaxis,...], (hits_test[14,:,:,:])[tf.newaxis,...], verbose=1);
# def hit_average_order2(data, y_pred):
#     y_hits_in_order2 = data[14,:,:,0]*data[14,:,:,3]
#     nofHitsInOrder2 = tf.math.count_nonzero(tf.greater(y_hits_in_order2,0.01), dtype=tf.float32)
#     print(nofHitsInOrder2)
#     return (K.sum(y_hits_in_order2*y_pred[14,:,:,0])/nofHitsInOrder2), nofHitsInOrder2

# or2, nofhitsin02 = hit_average_order2(hits_test, encoded )
# print(or2)
# print(nofhitsin02)
# # %%
# print(tf.math.count_nonzero(tf.greater(hits_test[0,:,:,2],0.), dtype=tf.float32))
# #tf.greater(hits_test[0,:,:,2],0.5)
# #tf.print(hits_test[0,:,:,2], summarize=-1)
# #print(hits_test[0,:,:,2])
# # %%

# %%
