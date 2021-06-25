#Species Classification Model

# Set up Tensorflow 2.0

from __future__ import absolute_import, division, print_function, unicode_literals

#try:
#  %tensorflow_version 2.x
#except Exception:
#  pass

import tensorflow as tf

# Import libraries

# General
import csv
import os
import numpy as np
from collections import defaultdict
import random
random.seed(42)

# Pretrained Models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16Pre
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as XceptionPre
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as MNPre
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as ResPre
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19Pre
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as DensePre
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input as LargePre

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split

# Model Configuration
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam

# Model Training
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# Visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%matplotlib inline

# Others
from keras.preprocessing import image as KImage
#from keras import backend as K

#K.tensorflow._get_available_gpus()

# Initial model configuration setting based on the value given in the variable "BASE_MODEL"
BASE_MODEL="resnet"  #Choices are: vgg16, mobilenetv2, xception

if BASE_MODEL=='vgg16':
  train_imagefile="Training-Images-224.csv"
  train_labelfile="Training-Labels-224.txt"
  test_imagefile="Test-Images-224.csv"
  test_labelfile="Test-Labels-224.txt"
  input_shape=(224,224,3)
  zero_model=VGG16(weights='imagenet',include_top=False,input_shape=input_shape,)
  saved_model_filename='species_classification_vgg16_model.h5'
  preprocessor=VGG16Pre

if BASE_MODEL=="vgg19":
    train_imagefile="Training-Images-VGG19-224.csv"
    train_labelfile="Training-Labels-VGG19-224.txt"
    test_imagefile="Test-Images-VGG19-224.csv"
    test_labelfile="Test-Labels-VGG19-224.txt"
    input_shape=(224,224,3)
    zero_model =VGG19(include_top=False, weights='imagenet', input_shape=input_shape,) 
    saved_model_filename='species_classification_vgg19_model.h5'
    preporcessor=ResPre

if BASE_MODEL=="densenet":
    train_imagefile="Training-Images-Dense-224.csv"
    train_labelfile="Training-Labels-Dense-224.txt"
    test_imagefile="Test-Images-Dense-224.csv"
    test_labelfile="Test-Labels-Dense-224.txt"
    input_shape=(224,224,3)
    zero_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape,)
    saved_model_filename='species_classification_dense_model.h5'
    preporcessor=DensePre

elif BASE_MODEL=='resnet':
    train_imagefile="Training-Images-res-224.csv"
    train_labelfile="Training-Labels-res-224.txt"
    test_imagefile="Test-Images-res-224.csv"
    test_labelfile="Test-Labels-res-224.txt"
    input_shape=(224, 224, 3)
    zero_model=ResNet50(weights='imagenet', include_top=False, input_shape=input_shape,)
    saved_model_filename='species_classification_resnet50_model.h5'
    preprocessor=ResPre

elif BASE_MODEL=="mobilenetv2":
  train_imagefile="Train-Images-Mobile-224.csv"
  train_labelfile="Train-Labels-Mobile-224.txt"
  test_imagefile="Test-Images-Mobile-224.csv"
  test_labelfile="Test-Labels-Mobile-224.txt"
  input_shape=(224,224,3)
  zero_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=input_shape)
  saved_model_filename='species_classification_mobilenetv2_model.h5'
  preprocessor=MNPre

elif BASE_MODEL=="xception":
  train_imagefile="Training-Images-Xception-224.csv"
  train_labelfile="Training-Labels-Xception-224.txt"
  test_imagefile="Test-Images-Xception-224.csv"
  test_labelfile="Test-Labels-Xception-224.txt"
  input_shape=(224,224,3)
  zero_model=Xception(weights='imagenet',include_top=False,input_shape=input_shape)
  saved_model_filename='species_classification_xception_model.h5'
  preprocessor=XceptionPre

elif BASE_MODEL=="nasnetlarge":
  train_imagefile="Training-Images-331.csv"
  train_labelfile="Training-Labels-331.txt"
  test_imagefile="Test-Images-331.csv"
  test_labelfile="Test-Labels-331.txt"
  input_shape=(331,331,3)
  zero_model=Xception(weights='imagenet',include_top=False,input_shape=input_shape)
  saved_model_filename='species_classification_nasnetlarge_model.h5'
  preprocessor=LargePre
  
# Set up path for csv files containing preprocessed images.
# 해당 csv파일이 있는 경로에 맞춰서 변수값을 정의한다.
csvpath='/mnt/data/guest1/crop_images/csv'

# Function to load processed image data in csv files (both training and test data and their corresponding labels)
def LoadData(train_imagefile=train_imagefile, train_labelfile=train_labelfile,
             test_imagefile=test_imagefile, test_labelfile=test_labelfile, input_shape=input_shape):
  # Training Dataset
  X=[]
  Individuals=[]
  Species=[]

  dataset=np.loadtxt(os.path.join(csvpath,train_imagefile),delimiter=",")
  f=open(os.path.join(csvpath,train_labelfile),'r')
  lines=f.readlines()
  for line in lines:
      vals=line.rstrip()
      Species.append(vals.split("-")[0])
      Individuals.append(vals)

  i=0
  for x in dataset:
    image=x.reshape(input_shape)
    X.append(image)

  # Test Dataset
  X_Test=[]
  Individuals_Test=[]
  Species_Test=[]
  dataset=np.loadtxt(os.path.join(csvpath,test_imagefile),delimiter=",")

  for x in dataset:
    image=x.reshape(input_shape)
    X_Test.append(image)

  f=open(os.path.join(csvpath,test_labelfile),'r')
  lines=f.readlines()
  
  for line in lines:
    vals=line.rstrip()
    Species_Test.append(vals.split("-")[0])
    Individuals_Test.append(vals)

  X_Test=np.asarray(X_Test)
  X=np.asarray(X)
  return(X,Species,Individuals,X_Test,Species_Test,Individuals_Test)

# Load pre-processed images

X,Species,Individuals,X_Test,Species_Test,Individuals_Test=LoadData(train_imagefile=train_imagefile, train_labelfile=train_labelfile,
               test_imagefile=test_imagefile, test_labelfile=test_labelfile, input_shape=input_shape)
print(X.shape)
# Use this line for augmented images
#X,Species,Individuals,Ind_DB,X_Test,Species_Test,Individuals_Test=LoadData(train_imagefile="Training-Images-224.csv",train_labelfile="Training-Labels-224.txt", test_imagefile="Test-Images-224.csv",test_labelfile="Test-Labels-224.txt")


#X,Species,Individuals,X_Test,Species_Test,Individuals_Test=LoadData(train_imagefile="Training-Images-224.csv",train_labelfile="Training-Labels-224.txt", test_imagefile="Test-Images-224.csv",test_labelfile="Test-Labels-224.txt")

# Label encoding
le = LabelEncoder()
le.fit(Species)
Y=le.transform(Species)
Y_Test=le.transform(Species_Test)
Y1=to_categorical(np.array(Y))
Y_Test1=to_categorical(np.array(Y_Test))
print(X.shape)
print(Y1.shape)
print(le.classes_)
num_species=len(le.classes_)

fp=open(os.path.join(csvpath,"species_list.pickle"), "wb")

# Train & Test data split
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X, Y1, test_size=0.10, random_state=42)

len(X_Train)
len(Y_Train)

tf.keras.backend.clear_session()
#vgg=VGG16(weights='imagenet',include_top=False,input_shape=input_shape)

resnet50=ResNet50(weights='imagenet', include_top=False, input_shape=input_shape) # resnet50모델을 사용하기 위해 해당 모델만 주석처리를 하지 않았다.

#densenet=DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

#vgg19=VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

#mobilenet=MobileNetV2(weights='imagenet',include_top=False,input_shape=input_shape)

#xception=Xception(weights='imagenet',include_top=False,input_shape=input_shape)


#Set all layers of pretrained VGG16 model as trainable. Add a few dense layers on top
zero_model.trainable=True 

spec_model=Sequential()
spec_model.add(zero_model)
spec_model.add(Flatten())
spec_model.add(Dropout(0.2))
spec_model.add(Dense(256, activation='relu',name="Embedding"))
spec_model.add(Dropout(0.1))
spec_model.add(Dense(128, activation='relu'))
spec_model.add(Dropout(0.1))
spec_model.add(Dense(64, activation='relu'))
spec_model.add(Dropout(0.1))
spec_model.add(Dense(num_species))

spec_model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint(os.path.join(csvpath,saved_model_filename), monitor='val_loss', mode='min')

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
spec_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
                   loss=loss_fn,
                   metrics=['accuracy'])

#STATIC
spec_model.fit(X_Train,Y_Train,validation_data=(X_Val,Y_Val),batch_size=8,epochs=90,verbose=2,callbacks=[es,mc])

#DYNAMIC ** Note: tried Validation without augmentation (from above) and got ~20% accuracy..
#history = vgg_model.fit_generator(training_generator,steps_per_epoch=(len(X_Train))//32, validation_data=validation_generator,validation_steps=len(X_Val)//32,epochs=30)

# Evaluate on test data WITHOUT augmentation
spec_model.evaluate(X_Test, Y_Test1, verbose=2)

# Accuracy breakdown by species
predictions = spec_model.predict(X_Test)
for idx, species in enumerate(le.classes_):
  total_count = 0
  correct_count = 0
  result_idx = 0
  for row in Y_Test1:
    if row[idx] == 1:
      total_count += 1
      predicted_value = np.argmax(predictions[result_idx]).item()
      if predicted_value == idx:
        correct_count += 1
    result_idx += 1
  print("Accuracy of "+species+" =",correct_count/total_count*100,"%")
  print(" Total =",total_count,", Correct Predictions =",correct_count)

# Create a prediction result that aligns with the actual target values instead of probabilities
pred_new = []
arr_len = len(le.classes_)
for idx in predictions:
  new_array = [0 for i in range(arr_len)]
  max_idx = np.argmax(idx)
  new_array[max_idx] = 1
  pred_new.append(new_array)
pred_new = np.asarray(pred_new)
print(pred_new.shape)
print(Y_Test1.shape)

# Create a confusion matrix
df_confusion = []
for num in range(arr_len):
  df_confusion.append([0] * arr_len)

df_confusion = np.asarray(df_confusion)

for pred, act in zip(pred_new, Y_Test1):
  pred_idx = np.where(pred == 1)[0][0]
  act_idx = np.where(act == 1)[0][0]
  df_confusion[act_idx][pred_idx] += 1
df_confusion

# Plot a confusion matrix with heatmap  
fig = plt.figure(figsize=(15, 15))
ax= fig.add_subplot()
sns.heatmap(df_confusion, annot=True, ax = ax, cmap="YlGnBu") #annot=True to annotate cells

# Add labels, title, and ticks
ax.set_xlabel('Predicted Species');ax.set_ylabel('Actual Species')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(le.classes_); ax.yaxis.set_ticklabels(le.classes_)

# Save plot
fig.savefig('/mnt/data/guest1/crop_images/Presentations/species_classification_confusion_matrix.png')

spec_model.load_weights(os.path.join(csvpath,saved_model_filename))
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
spec_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
                   loss=loss_fn,
                   metrics=['accuracy'])

#test_generator = datagen.flow(X_Test, Y_Test1, batch_size=32,shuffle=True,seed=7)
spec_model.evaluate(X_Test,  Y_Test1, verbose=2)

len(X_Test)

def plotprints(df):
  dfx=df.drop(['y','Names'],axis=1)
  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
  tsne_results = tsne.fit_transform(dfx)
  df['tsne-2d-one'] = tsne_results[:,0]
  df['tsne-2d-two'] = tsne_results[:,1]
  plt.figure(figsize=(16,10))
  num=df['Names'].nunique()

  sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                  hue="Names",
                  palette=sns.color_palette("hls", num),
                  data=df,
                  legend="full",
                  alpha=0.6)

Y=le.transform(Species)
X_encoded=spec_model.predict(X)
df=pd.DataFrame(X_encoded)
df['y']=Y
df['Names']=Species
# Plot projected 2D clusters for each species
plotprints(df)

# Set all layers of pretrained VGG16 model as trainable. Add a few dense layers on top.
trained_model=Sequential()
trained_model.add(zero_model)
trained_model.add(Flatten())
trained_model.add(Dropout(0.4))
trained_model.add(Dense(256, activation='relu',name="Dense1"))
trained_model.add(Dense(128, activation='relu'))
trained_model.add(Dense(64, activation='relu'))
trained_model.add(Dropout(0.4))
trained_model.add(Dense(num_species))

trained_model.summary()

trained_model.load_weights(os.path.join(csvpath,saved_model_filename))
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
trained_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss=loss_fn,
                      metrics=['accuracy'])

# Load Data
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
    #x=x.reshape(-1)
    return x

def LoadDataSet(folder,preprocessor,size=(224,224)):
  Prints=[]
  Instances=[]
  Individuals=[]
  for instance in os.listdir(folder):
    ind_path=folder+'/'+instance
    for footprint in os.listdir(ind_path):
      x=load_image(os.path.join(ind_path,footprint),preprocessor,(224,224))
      if x.shape[0]==0:
        continue
      else:
        Prints.append(x)
        Instances.append(instance)
        Individuals.append(footprint)
  return Prints,Instances,Individuals

#inferencepath="/mnt/data/guest1/crop_images/Test_Data"
#Prints,Instances,Individuals=LoadDataSet(inferencepath,preprocessor,(224,224))

X=np.asarray(Prints)
#X=np.array(Prints)
Y_logits=np.absolute(trained_model.predict(X))
Y=np.argmax(Y_logits,1)
Y_Species=le.inverse_transform(Y)

Y_logits[0]

for i in range(len(Prints)):
  print(Y_Species[i],Individuals[i])
