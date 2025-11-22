#!/usr/bin/env python
# coding: utf-8

# Noah Harrison 2046687

# In[1]:


# libraries that might be helpful
import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from os import listdir
from skimage.transform import rotate
import random
import tensorflow as tf


# In[2]:


SEED = 42
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# In[3]:


def get_classes(filedir):
    f = open(filedir)
    lines = f.read().splitlines()
    lines.sort()
    vector = np.zeros(len(lines))
    for i in range(len(lines)):
        vector[i] = lines[i].split("\t")[1]
    return vector


# In[4]:


def get_images(directory, img_shape):
    info = os.listdir(directory)
    info.sort()
    vector = np.zeros(shape=(len(info), img_shape[0], img_shape[1], img_shape[2]))
    for i in range(len(vector)):
        x = io.imread(f"{directory}{info[i]}")
        vector[i,:,:,:] = x
    return vector


# In[5]:


def shuffle_data(x, y):
    ind = np.arange(len(x))
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]
    return x, y


# In[6]:


def augment_images(images, labels, im_to_gen = 2):
    no_images = len(images)
    print(no_images)
    for i in range(no_images):
        if i%100 == 0:
            print(i)
            print(images.shape)
        for k in range(im_to_gen):
            augmented_image = skimage.transform.rotate(images[i,:,:,:], angle=random.randint(-45,45))
            augmented_image = np.expand_dims(augmented_image, axis=0)
            images = np.vstack((images, augmented_image))
            labels = np.append(labels, labels[i])
    return images, labels


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, SpatialDropout2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, SGD
def construct_model(img_shape, lr):
    model = Sequential()
    model.add( Conv2D(32, kernel_size=(3,3), input_shape=img_shape) )
    model.add( Activation("relu") )
    model.add( MaxPooling2D(pool_size=(2,2)) )
    model.add( Dropout(0.2) )
    model.add( Conv2D(64, kernel_size=(3,3) ) )
    model.add( Activation("relu") )
    model.add( MaxPooling2D(pool_size=(2,2)) )
    model.add( Dropout(0.3) )
    model.add( Flatten() )
    model.add( Dense(100) )
    model.add( Activation("relu") )
    model.add( Dropout(0.5) )
    model.add( Dense(1) )
    model.add( Activation("sigmoid") )
    model.compile( loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'] )
    return model


# In[8]:


import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
def train_test_model(img_shape, lr, x_train, y_train, x_test, y_test, x_val, y_val, batch_size, results, data_used, i):
    model = construct_model(img_shape, lr)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    training = model.fit(x_train, y_train, epochs=100, verbose=2, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[early_stopping_monitor])
    val_loss, val_acc = model.evaluate(x_val, y_val)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    new_row = [data_used, lr, batch_size, val_loss, val_acc, test_loss, test_acc, i]
    results.loc[len(results)] = new_row
    return model, results


# In[9]:


def plotHistory(Tuning):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(Tuning.history['loss'])
    axs[0].plot(Tuning.history['val_loss'])
    axs[0].set_title('loss vs epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'vali'], loc='upper left')
    
    axs[1].plot(Tuning.history['accuracy'])
    axs[1].plot(Tuning.history['val_accuracy'])
    axs[1].set_title('accuracy vs epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylim([0.0,1.0])
    axs[1].legend(['train', 'vali'], loc='upper left')
    plt.show(block = False)
    plt.show()


# In[10]:


# get sample image
img = io.imread(f"cat_dog/train/{os.listdir('cat_dog/train')[0]}")
io.imshow(img)
plt.show()


# In[11]:


# store class labels
y_train = get_classes("cat_dog/train_class_labels.txt")
y_test = get_classes("cat_dog/test_class_labels.txt")
# store images
x_train = get_images("cat_dog/train/", img.shape)
x_train = x_train / 255
x_test = get_images("cat_dog/test/", img.shape)
x_test = x_test / 255
# cast classes to int
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")
# shuffle the images
x_train, y_train = shuffle_data(x_train, y_train)
x_test, y_test = shuffle_data(x_test, y_test)
# create a validation set
val_index = len(x_train) - int( len(x_train) * 0.2 )
x_val = x_train[val_index:]
y_val = y_train[val_index:]
x_train = x_train[:val_index]
y_train = y_train[:val_index]


# In[12]:


# augment images
x_train_augment, y_train_augment = augment_images(np.copy(x_train), np.copy(y_train), 2)


# In[13]:


# demonstration of image augmentation
for i in range(1, 5 + 1):
    plt.subplot(2, 5, i)
    random_img = random.randint(0, 8000)
    plt.imshow(x_train[random_img])
    plt.axis('off')
    plt.subplot(2, 5, 5 + i)
    plt.imshow(x_train_augment[8000 + (random_img * 2)])
    plt.axis('off')


# In[14]:


results_nonaugmented = pd.DataFrame(columns=['dataset', 'lr', 'batch_size', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'i'])
results_augmented = pd.DataFrame(columns=['dataset', 'lr', 'batch_size', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'i'])
models = []
i = 0
for rate in [0.001, 0.005, 0.05]:
    for batch_size in [50, 100, 200, 500, 1000]:
        x_train, y_train = shuffle_data(x_train, y_train)
        x_train_augment, y_train_augment = shuffle_data(x_train_augment, y_train_augment)
        model, results_nonaugmented = train_test_model(img.shape, rate, x_train, y_train, x_test, y_test, x_val, y_val, batch_size, results_nonaugmented, 'original', i)
        models.append(model)
        i += 1
        model, results_augmented = train_test_model(img.shape, rate, x_train_augment, y_train_augment, x_test, y_test, x_val, y_val, batch_size, results_augmented, 'augmented', i)
        models.append(model)
        i += 1
results = pd.concat([results_nonaugmented, results_augmented])
results_augmented = results_augmented.sort_values(by=['test_acc', 'test_loss'], ascending=False).reset_index()
results_nonaugmented = results_nonaugmented.sort_values(by=['test_acc', 'test_loss'], ascending=False).reset_index()
results = results.sort_values(by=['test_acc', 'test_loss'], ascending=False).reset_index()


# In[15]:


# total results
results.iloc[:,1:-1]


# In[16]:


# results for models trained on non-augmented data only
results_nonaugmented.iloc[:,1:-1]


# In[17]:


# results for models trained on augmented data
results_augmented.iloc[:,1:-1]


# In[18]:


best = models[ int(results_nonaugmented["i"][0]) ]
best_augmented = models[ int(results_augmented["i"][0]) ]


# In[19]:


# optimization history of best non-augmented model
plotHistory(best.history)


# In[20]:


# optimization history of best augmented model
plotHistory(best_augmented.history)

