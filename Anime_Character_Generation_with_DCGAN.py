#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install PyDrive')




import os 
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials


# In[ ]:


#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)


# In[ ]:


#download = drive.CreateFile({'id': '0B4wZXrs0DHMHMEl1ODVpMjRTWEk'})
#download.GetContentFile('anime-faces.tar.gz')


# In[ ]:


#ls


# In[ ]:


#get_ipython().system('tar -xvzf anime-faces.tar.gz')


# In[6]:


#get_ipython().system('cd anime-faces')









#get_ipython().system('rm config.json annotations.csv')


# In[8]:


#get_ipython().system('rm -v ._.DS_Store .DS_Store')


# In[ ]:




# In[9]:


#get_ipython().system('cd ..')


# In[ ]:


#get_ipython().system('mkdir results')


# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
import os
from PIL import Image

import tensorflow as tf
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, Conv2DTranspose, Activation, UpSampling2D, Dropout, ReLU, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras_preprocessing import image
from keras.models import load_model, Sequential
from keras.utils import multi_gpu_model
import keras.backend as K

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)


def load_images(src):
  #src= 'anime-faces'
  image_data = []
  count = 0
  for file in os.listdir(src):
    file_path = os.path.join(src, file)
    for image in os.listdir(file_path):
      if not image.startswith('.'):
        img_path = os.path.join(file_path, image)    
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        #print(img.shape)
        img = (img-127.5)/127.5
        image_data.append(img)
        count = count + 1
        #print(len(image_data))
        
        if count > 50000:
          #print("Break")
          break
          
  image_data = np.asarray(image_data)
  #print(image_data.shape)
  #image_data = (image_data - 127.5)/127.5
  return(image_data)
          


# In[ ]:


def build_gen():
  
  latent_dims = 100
  #gen_model = Sequential()

  inputs = Input(shape = (latent_dims,))

  x = Dense(units = 2048)(inputs)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = Dense(256 * 8 * 8)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = Reshape((8, 8, 256), input_shape = (256 * 8 * 8,))(x)
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(128, (5, 5), padding = 'same')(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(64, (5, 5), padding = 'same')(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(3, (5, 5), padding = 'same')(x)
  x = ReLU()(x)

  gen_model = Model(input = [inputs], outputs = [x])

  #gen_model.add(Dense(units = 2048))
  #gen_model.add(LeakyReLU(alpha = 0.2))
  
  #gen_model.add(Dense(256 * 8 * 8))
  #gen_model.add(BatchNormalization())
  #gen_model.add(LeakyReLU(alpha = 0.2))
  
  #gen_model.add(Reshape((8, 8, 256), input_shape = (256 * 8 * 8,)))
  #gen_model.add(UpSampling2D(size = (2, 2)))
  #gen_model.add(Conv2D(128, (5, 5), padding = 'same'))
  #gen_model.add(LeakyReLU(alpha = 0.2))
  
  #gen_model.add(UpSampling2D(size = (2, 2)))
  #gen_model.add(Conv2D(64, (5, 5), padding = 'same'))
  #gen_model.add(LeakyReLU(alpha = 0.2))
  
  #gen_model.add(UpSampling2D(size = (2, 2)))
  #gen_model.add(Conv2D(3, (5, 5), padding = 'same'))
  #gen_model.add(ReLU())
  
  return gen_model


# In[ ]:


def build_disc():
  
  #disc_model = Sequential()
  
  input_shape = (64, 64, 3)
  inputs = Input(shape = input_shape)

  x = Conv2D(filters = 128 , kernel_size = 5, padding = 'same')(inputs)
  x = LeakyReLU(alpha = 0.2)(x)
  x = MaxPooling2D(pool_size = (2, 2))(x)
  
  x = Conv2D(filters = 512 , kernel_size = 5, padding = 'same')(x)
  x = LeakyReLU(alpha = 0.2)(x)
  x = MaxPooling2D(pool_size = (2, 2))(x)
  
  x = Flatten()(x)
  
  x = Dense(1024)(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = Dense(1)(x)
  x = Activation('sigmoid')(x)
  
  disc_model = Model(inputs = [inputs], outputs = [x])



  #disc_model.add(Conv2D(128 , (5,5), padding = 'same', input_shape = (64, 64, 3)))
  #disc_model.add(LeakyReLU(alpha = 0.2))
  #disc_model.add(MaxPooling2D(pool_size = (2, 2)))
  
  #disc_model.add(Conv2D(512 , (5,5), padding = 'same', input_shape = (64, 64, 3)))
  #disc_model.add(LeakyReLU(alpha = 0.2))
  #disc_model.add(MaxPooling2D(pool_size = (2, 2)))
  
  #disc_model.add(Flatten())
  
  #disc_model.add(Dense(1024))
  #disc_model.add(LeakyReLU(alpha = 0.2))
  
  #disc_model.add(Dense(1))
  #disc_model.add(Activation('sigmoid'))
  return disc_model


# In[ ]:


def save_rgb_img(img, path):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


# In[ ]:


def train():
  
  start_time = time.time()
  epochs = 10000
  batch_size = 128
  z_shape = 100
  lr = 0.0005
  momentum = 0.9
  
  
  print("Loading Images")
  
  X = load_images('anime-faces')
  print(X.shape)
  
  print("Images Loaded")
  
  #X = (X-127.5)/127.5
  
  print("Normalized Images")
  
  disc_optimizer = SGD(lr = lr, momentum = momentum, nesterov = True)
  gen_optimizer = SGD(lr = lr, momentum = momentum, nesterov = True)
  
  gen_model = build_gen()
  print(gen_model.outputs)
  gen_model = multi_gpu_model(gen_model, gpus = 3)
  gen_model.compile(loss = 'binary_crossentropy', optimizer = gen_optimizer)
  
  disc_model = build_disc()
  print(disc_model.outputs)
  disc_model = multi_gpu_model(disc_model, gpus = 3)
  disc_model.compile(loss = 'binary_crossentropy', optimizer = disc_optimizer)
  
  #adversarial_model = Sequential()
  #adversarial_model.add(gen_model)
  disc_model.trainable = False
  #adversarial_model.add(disc_model)

  input_z_noise = Input(shape = (100,))
  recons_image = gen_model([input_z_noise])
  valid = disc_model([recons_image])

  adversarial_model = Model(inputs = [input_z_noise], outputs = [valid])
  
  adversarial_model = multi_gpu_model(adversarial_model, gpus = 3)
  adversarial_model.compile(loss = 'binary_crossentropy', optimizer = gen_optimizer)
  
  #tensorboard = TensorBoard(log_dir = "logs/{}".format(time.time()), write_images = True, write_grads = True , write_graphs = True)
  #tensorboard.set_model(gen_model)
  #tensorboard.set_model(disc_model)
  
  for epoch in range(epochs):
    
    print("Epoch: ", epoch)
    
    number_of_batches = int(X.shape[0] / batch_size)
    
    print("Number of Batches: ", number_of_batches)
    
    for index in range(number_of_batches):
      
      #print("Batch: ", index)
      
      # Training the generator
      
      z_noise = np.random.normal(-1, 1, size = (batch_size, z_shape))
      
      image_batch = X[index * batch_size : (index + 1) * batch_size]
      
      generated_images = gen_model.predict_on_batch(z_noise)
      
      y_real = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
      
      y_fake = np.random.random_sample(batch_size) * 0.2
      
      disc_loss_real = disc_model.train_on_batch(image_batch, y_real)
      disc_loss_fake = disc_model.train_on_batch(generated_images, y_fake)
      
      disc_loss = (disc_loss_real + disc_loss_fake)/2
      
      
      
      # Training the adversarial
      
      z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
      
      gen_loss = adversarial_model.train_on_batch(z_noise, y_real)
      
    print("Generator Loss: ", gen_loss)
    print("Discriminator Loss: ", disc_loss)
      
    if epoch % 10 == 0:

        
      z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
      gen_images1 = gen_model.predict_on_batch(z_noise)
        
      for index, img in enumerate(gen_images1[:2]):
        save_rgb_img(img, "results/image_generated_{}_{}.png".format(epoch, index))
          
      gen_model.save("generator.h5")
      disc_model.save("discriminator.h5")
        
        
  print("Time:", (time.time() - start_time))


# In[ ]:


if __name__ == '__main__':
  train()


# In[ ]:




