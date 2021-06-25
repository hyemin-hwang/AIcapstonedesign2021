#Individual identification

#Set up Tensor flow 2.0

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf


#General
import cv2
import csv
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import defaultdict
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import pickle

# TF2.0/ Keras Libraries

from tensorflow.keras import backend as K

# Keras Imagenet pre=-trained models and pre-processors
from keras.preprocessing import image as KImage
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16Pre

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as ResPre

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19Pre

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as XceptionPre

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as MNPre

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as DensePre

from tensorflow.keras.applications.imagenet_utils import preprocess_input


# TF2/ Keras Modeling utilities
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import glorot_normal

# Plotting/ Visualization

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

random.seed(42)

#Set up various paths
#Set up path for csv files containing preprocessed images. CHange subfolder names to match your setup in google drive
csvpath='/mnt/data/guest1/crop_images/csv'
path="/mnt/data/guest1/crop_images/Training_Data"
test_path=" /mnt/data/guest1/crop_images/Test_Data"
modelpath='/mnt/data/guest1/crop_images/csv'

BASE_MODEL="resnet"  #CHoices are: vgg16, mobilenetv2, Xception

if BASE_MODEL=='vgg16':
  train_imagefile="Training-Images-224.csv"
  train_labelfile="Training-Labels-224.txt"
  test_imagefile="Test-Images-224.csv"
  test_labelfile="Test-Labels-224.txt"
  input_shape=(224,224,3)
  pretrained_model='species_classification_vgg16_model.h5'
  preprocessor=VGG16Pre
  savefile='vgg16_best_model'
  savemodel='vgg16_best_model.h5'

elif BASE_MODEL=="mobilenetv2":
  train_imagefile="Train-Images-Mobile-224.csv"
  train_labelfile="Train-Labels-Mobile-224.txt"
  test_imagefile="Test-Images-Mobile-224.csv"
  test_labelfile="Test-Labels-Mobile-224.txt"
  input_shape=(224,224,3)
  pretrained_model='species_classification_mobilenetv2_model.h5'
  preprocessor=MNPre
  savefile='mobilenetv2_best_model'

elif BASE_MODEL=="xception":
  train_imagefile="Training-Images-Xception-224.csv"
  train_labelfile="Training-Labels-Xception-224.txt"
  test_imagefile="Test-Images-Xception-224.csv"
  test_labelfile="Test-Labels-Xception-224.txt"
  input_shape=(224,224,3)
  pretrained_model='species_classification_xception_model.h5'
  preprocessor=XceptionPre
  savefile='xception_best_model'

elif BASE_MODEL=='resnet':
  train_imagefile="Training-Images-res-224.csv"
  train_labelfile="Training-Labels-res-224.txt"
  test_imagefile="Test-Images-res-224.csv"
  test_labelfile="Test-Labels-res-224.txt"
  input_shape=(224,224,3)
  pretrained_model='species_classification_resnet101V2_model.h5'
  preprocessor=ResPre
  savefile='resnet101V2_best_model'
  savemodel='resnet101V2_best_model.h5'

elif BASE_MODEL=="vgg19":
    train_imagefile="Training-Images-VGG19-224.csv"
    train_labelfile="Training-Labels-VGG19-224.txt"
    test_imagefile="Test-Images-VGG19-224.csv"
    test_labelfile="Test-Labels-VGG19-224.txt"
    input_shape=(224,224,3)
    pretrained_model='species_classification_vgg19_model.h5'
    preporcessor=VGG19Pre
    savefile='vgg19_best_model'
    savemodel='vgg19_best_model.h5'

elif BASE_MODEL=="densenet":
  train_imagefile="Training-Images-Dense-224.csv"
  train_labelfile="Training-Labels-Dense-224.txt"
  test_imagefile="Test-Images-Dense-224.csv"
  test_labelfile="Test-Labels-Dense-224.txt"
  input_shape=(224,224,3)
  pretrained_model='species_classification_dense_model.h5'
  preprocessor=DensePre
  savefile='dense_best_model'
  savemodel='dense_best_model.h5'


  #Function to load processed image data in csv files (both training and test, input data labels)
def LoadData(train_imagefile=train_imagefile,train_labelfile=train_labelfile,
             test_imagefile=test_imagefile,test_labelfile=test_labelfile):
  #Training Data Set
  Ind_DB=defaultdict(defaultdict)
  Individuals=[]
  Species=[]
  X=[]
  dataset=np.loadtxt(os.path.join(csvpath,train_imagefile),delimiter=",")
  f=open(os.path.join(csvpath,train_labelfile),'r')
  lines=f.readlines()
  for line in lines:
    vals=line.rstrip()
    Species.append(vals.split("-")[0])
    Individuals.append(vals)
  f.close()
    
  i=0
  for x in dataset:
    image=x.reshape(224,224,3)
    X.append(image)
    species=Species[i]
    key=Individuals[i]
    spec_DB=Ind_DB[species]
    if key not in spec_DB.keys():
      spec_DB[key]=[image]
    else:
      spec_DB[key].append(image)
    i=i+1


  #Test Data Set
  X_Test=[]
  Individuals_Test=[]
  Species_Test=[]
  dataset=np.loadtxt(os.path.join(csvpath,test_imagefile),delimiter=",")

  for x in dataset:
    image=x.reshape(224,224,3)
    X_Test.append(image)

  f=open(os.path.join(csvpath,test_labelfile),'r')
  lines=f.readlines()
  for line in lines:
    vals=line.rstrip()
    Species_Test.append(vals.split("-")[0])
    Individuals_Test.append(vals)
  f.close()

  X_Test=np.asarray(X_Test)
    
  return (Ind_DB,X_Test,Species_Test,Individuals_Test)

# Load Pre-Processed Images
Ind_DB,X_Test,Species_Test,Individuals_Test=LoadData(train_imagefile=train_imagefile, train_labelfile=train_labelfile,
                                                     test_imagefile=test_imagefile, test_labelfile=test_labelfile)

# Function create triples (A1,A2,B) for individuals within a species. All possible combinations are enumerated and written back to file

Species=[]

# Given a pair, add distinct footprint to create triples
def UpdateTriples(doubles,footprint):
  new_triples=[]
  for double in doubles:
    if double[0]==footprint or double[1]==footprint:
      print("Error in Update: ",double,footprint)
    new_triples.append((double[0],double[1],footprint))
  return new_triples

def AddTriples(singles,previous,footprint):
  new_triples=[]
  new_doubles=[]

  for base in previous:
    new_doubles.append((base,footprint))
    for single in singles:
      if base==single or footprint==single:
        print("Error in add: ",base,footprint,single)
      new_triples.append((base,footprint,single))
  return new_doubles,new_triples


# Function to go through each species and generate triples 
# that are then written back out to file. 

def LoadDataSet(DB,output_folder,outfile):
  for species in DB.keys():

    Species.append(species)
    filename=outfile+"_"+species+".csv"
    f = open(os.path.join(output_folder,filename),"w")
    writer = csv.writer(f)

    print("\n\n*** SPECIES:  ",species)

    triples=[]
    doubles=[]
    singles=[]
    individuals=DB[species]

    for individual,printlist in individuals.items():
      print("\n* INDIVIDUAL: ",individual)
      previous=[]
      prev_doubles=[]
      num_prints=len(printlist)

      for i in range(num_prints):
        uniq_print=individual+'|'+str(i)
        new_triples1=UpdateTriples(doubles,uniq_print)
        new_doubles,new_triples2=AddTriples(singles,previous,uniq_print)
        prev_doubles.extend(new_doubles)
        for triple in new_triples1:
          writer.writerow(triple)
        for triple in new_triples2:
          writer.writerow(triple)
        previous.append(uniq_print)
      doubles.extend(prev_doubles)
      singles.extend(previous)
    f.close()

# Create Triples
LoadDataSet(Ind_DB,csvpath,"triples")

#SPlit out Test vs Validation Data for triples
from sklearn.model_selection import train_test_split
for species in Species:
  fname='triples_'+species+'.csv'
  dataset=pd.read_csv(os.path.join(csvpath,fname),header=None)
  trainset,devset = train_test_split(dataset, test_size=0.25, random_state=42)
  fname='triples_'+species+'_train.csv'
  trainset.to_csv(os.path.join(csvpath,fname))
  fname='triples_'+species+'_dev.csv'
  devset.to_csv(os.path.join(csvpath,fname))

#Create Label Encoding for Species
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
le = LabelEncoder()
le.fit(Species)
Y=le.transform(Species)
print(le.classes_)

# Function returns image array given species and key

def get_img(DB,species,key):
  #print(key)
  pipe='|'
  values=key.split(pipe)
  individual=values[0]
  indx=int(values[1])
  try:
    spec_DB=DB[species]
    imglist=spec_DB[individual]
    x=imglist[indx]
  except:
    print("Error with loading ",key)  
    x=np.zeros((224,224,3))

  return x

from keras import backend as K
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall
def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision

def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score

# Model Generator class to generate triple sets of images for a given species, 
# given batch size adn number of steps.

def triples_generator(folder,DB,species,dataset="train",batch_size=4,num_steps=100):
  fname='triples_'+str(species)+'_'+dataset+'.csv'
  df=pd.read_csv(os.path.join(folder,fname))
  target=np.zeros((batch_size,768))
  total=df.shape[0]
  sample_size=int(num_steps*batch_size)

  while 1:
    indices=np.random.randint(0,total,size=sample_size)

    for i in range(num_steps):
      triples=[np.zeros((batch_size,224,224,3))for i in range(3)]
      cnt=0

      for j in range((i*batch_size),((i+1)*batch_size)):
        k=indices[j]
        triples[0][cnt,:,:,:]=get_img(DB,species,df.iloc[k,1])
        triples[1][cnt,:,:,:]=get_img(DB,species,df.iloc[k,2])
        triples[2][cnt,:,:,:]=get_img(DB,species,df.iloc[k,3])
        cnt=cnt+1

      yield (triples, target)


#CUstom loss function for Triplets Network
def triplet_loss(y_true,y_pred,alpha=1.2):
  ln=y_pred.shape.as_list()[-1]
  anchor=y_pred[:,0:int(ln/3)]
  positive=y_pred[:,int(ln/3):int(2*ln/3)]
  negative=y_pred[:,int(2*ln/3):ln]

  p_dist=K.sqrt(K.sum(K.square(anchor-positive),axis=1))
  n_dist=K.sqrt(K.sum(K.square(anchor-negative),axis=1))
  loss=K.maximum(p_dist-n_dist+alpha,0.0)
  return K.mean(loss)  

#Return L2 Norm
def calcl2(X,prints):
  l2norm=[]
  for i in range(len(prints)):
    l2norm.append(np.linalg.norm(X - prints[i]))
  return l2norm

#Function to create triplets model starting with a base model pre-trained as a species classifier

def Create_TripletTrainer(csvpath,pretrained_model=pretrained_model,input_shape=(224,224,3)):
  zero_model = load_model(os.path.join(csvpath,pretrained_model))
  x=zero_model.get_layer('Embedding').output
  x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
  triplet_model=Model(inputs=zero_model.input,outputs=x)
  input_shape=[224,224,3]
  X1=Input(input_shape)
  X2=Input(input_shape)
  X3=Input(input_shape)
  encoded1 = triplet_model(X1)
  encoded2 = triplet_model(X2)
  encoded3 = triplet_model(X3)

  concat_vector=concatenate([encoded1,encoded2,encoded3],axis=-1,name='concat')
  model=Model(inputs=[X1,X2,X3],outputs=concat_vector)
  model.compile(loss=triplet_loss,optimizer=Adam(0.000005))

  return model

# Function to Train a model for a given species using a pre-trained model as a base. 
# Trained model weights are saved to file passed into the savefile parameter. 

def TrainModel(DB,species="dog",pretrained_model=pretrained_model,input_shape=(224,224,3),savefile=savefile):
  tf.keras.backend.clear_session()

  model=Create_TripletTrainer(modelpath,pretrained_model,input_shape=(224,224,3))

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
  chkpoint=savefile+'_'+str(species)+'.h5'
  mc = ModelCheckpoint(os.path.join(modelpath,'testing',chkpoint), save_weights_only=True,monitor='val_loss', mode='min')
  train_gen=triples_generator(csvpath,DB,species,dataset="train",batch_size=4,num_steps=200)
  val_gen=triples_generator(csvpath,DB,species,dataset="dev",batch_size=4,num_steps=40)

  print("Training for Species: ",species)
  
  model.fit(train_gen,steps_per_epoch=300, epochs=30,verbose=1,validation_data=val_gen,validation_steps=40,callbacks=[es,mc])
  
  return True

TrainModel(Ind_DB,"dog")
