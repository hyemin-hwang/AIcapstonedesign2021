#reference: (WildTrack AI) https://github.com/jtdsouza/w251-WildTrackAI.git
#Data PreProcessing

#Set up TF 2.0
from __future__ import absolute_import, division, print_function, unicode_literals

#try:
 # %tensorflow_version 2.x
#except Exception:
#  pass

import tensorflow as tf

#Import Libraries

#General
import csv
import os
import numpy as np
from collections import defaultdict
import random
import pickle

# Image processing libraries
from keras.preprocessing import image as KImage
import cv2
#from google.colab.patches import cv2_imshow


# Preprocessors from Pretrained Models

from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16Pre
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19Pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as InceptionPre
from tensorflow.keras.applications.densenet import preprocess_input as DensePre
from tensorflow.keras.applications.xception import preprocess_input as XceptionPre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as MNPre
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as ResPre
from tensorflow.keras.applications.nasnet import preprocess_input as LargePre

# Plotting 
#%matplotlib inline
import matplotlib.pyplot as plt

#Set up path for training/ test data.
#README.md 파일에 있는 dataset이 저장된 경로에서 dataset을 가져온 후
#Training_Data, Test_Data 폴더가 들어 있는 폴더에 csv폴더와 Presentations폴더를 만들어 추가한다.
#그리고 csv폴더 안에 testing폴더를 새로 만들어 추가해준다.
#만든 폴더의 경로를 다음과 같은 경로를 나타내는 변수에 지정해준다.
path="/mnt/data/guest1/crop_images/Training_Data"
test_path="/mnt/data/guest1/crop_images/Test_Data"
csvpath="/mnt/data/guest1/crop_images/csv"
os.listdir(path)

# Function to load and individual image to a specified size.
# Pre-trained model에 맞춰 input image size를 정해주고 image를 load해준다.
def temp_load_image(image,preprocessor,size=(299,299)):
  try:
    print(image)
    image = KImage.load_img(image,target_size=size,interpolation="nearest")

  except:
    return np.zeros(0),np.zeros(0)
  else:
    raw= KImage.img_to_array(image)
    x = np.expand_dims(raw, axis=0)
    x = preprocessor(x)
    x=np.squeeze(x)
    #x=x.reshape(-1)
    return image,x

#Test One Image
raw,x=temp_load_image(os.path.join(test_path,'dog','mi','mi_02_crop_0.jpg'), MNPre,(224,224))
opencvImage = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
plt.imshow(raw)
plt.show()

# Function to load and individual image to a specified size.
def load_image(image,preprocessor,size=(224,224)):
  try:
    image = KImage.load_img(image,target_size=size,interpolation="nearest")
  except:
    return np.zeros(0)
  else:
    x= KImage.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocessor(x)
    x=np.squeeze(x)
    x=x.reshape(-1)
    return x

# Function to load image data set from specified folder. Will be reshaped to size specified in size parameter.
# Once images are processed, writes back out to a csv file
# csv file에는 이미지의 정보가 담긴다.(A PIL Image instance.)
def LoadDataSet(folder,img_output,label_output,preprocessor,size=(224,224)):
  Keys=[]
  with open(img_output, 'w') as f:
    writer = csv.writer(f)
    for species in os.listdir(folder):
      ind_path=folder+'/'+species
      print(species)

      for individual in os.listdir(ind_path):
        print_path=ind_path+'/'+individual

        for footprint in os.listdir(print_path):
          if footprint.find(" RH ")>0:
            continue
          else:
            x=load_image(os.path.join(print_path,footprint),preprocessor,(224,224)) #return값: A PIL Image instance.
            if x.shape[0]==0:
              continue
            else:
              key=species+"-"+individual
              Keys.append(key)
              writer.writerow(x)

  with open(label_output, 'w') as filehandle:
    filehandle.writelines("%s\n" % key for key in Keys)

#Point to path to write csv images and create pre-processed training and test input for VGG16 model (224x224 image size) 

LoadDataSet(path,os.path.join(csvpath,"Training-Images-res-224.csv"),os.path.join(csvpath,"Training-Labels-res-224.txt"),ResPre,(224,224))
LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-res-224.csv"),os.path.join(csvpath,"Test-Labels-res-224.txt"),ResPre,(224,224))

#Create similar csv files for Mobilenet (224x224) and NASNetLarge (331x331)
LoadDataSet(path,os.path.join(csvpath,"Training-Images-331.csv"),os.path.join(csvpath,"Training-Labels-331.txt"),preprocess_input,(331,331))
LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-331.csv"),os.path.join(csvpath,"Test-Labels-331.txt"),preprocess_input,(331,331))
LoadDataSet(path,os.path.join(csvpath,"Train-Images-Mobile-224.csv"),os.path.join(csvpath,"Train-Labels-Mobile-224.txt"),MNPre,(224,224))
LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-Mobile-224.csv"),os.path.join(csvpath,"Test-Labels-Mobile-224.txt"),MNPre,(224,224))

#csv files for Xception, VGG19 and Inception

#LoadDataSet(path,os.path.join(csvpath,"Training-Images-Xception-224.csv"),os.path.join(csvpath,"Training-Labels-Xception-224.txt"),XceptionPre,(299,299))
#LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-Xception-224.csv"),os.path.join(csvpath,"Test-Labels-Xception-224.txt"),XceptionPre,(299,299))
LoadDataSet(path,os.path.join(csvpath,"Training-Images-VGG19-224.csv"),os.path.join(csvpath,"Training-Labels-VGG19-224.txt"),ResPre,(224,224))
LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-VGG19-224.csv"),os.path.join(csvpath,"Test-Labels-VGG19-224.txt"),ResPre,(224,224))
#LoadDataSet(path,os.path.join(csvpath,"Training-Images-Inception-299.csv"),os.path.join(csvpath,"Training-Labels-Inception-299.txt"),InceptionPre,(299,299))
#LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-Inception-299.csv"),os.path.join(csvpath,"Test-Labels-Inception-299.txt"),InceptionPre,(299,299))
LoadDataSet(path,os.path.join(csvpath,"Training-Images-Dense-224.csv"),os.path.join(csvpath,"Training-Labels-Dense-224.txt"),DensePre,(224,224))
LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-Dense-224.csv"),os.path.join(csvpath,"Test-Labels-Dense-224.txt"),DensePre,(224,224))

#csv files for EfficientNET
#LoadDataSet(path,os.path.join(csvpath,"Training-Images-EfficientB4-224.csv"),os.path.join(csvpath,"Training-Labels-EfficientB4-224.txt"),XceptionPre,(299,299))
#LoadDataSet(test_path,os.path.join(csvpath,"Test-Images-EfficientB4-224.csv"),os.path.join(csvpath,"Test-Labels-EfficientB4-224.txt"),XceptionPre,(299,299))
