# AIcapstonedesign2021
AI capstone design 2021

The project was produced as the final project for the AI capstoneedesign 2021 class.

Participants are Lee Ho-jeong and Hwang Hye-min.

dataset>
  1. Cropped Images of Original Images (for One footprint) --> folder: crop_images
  2. augmented Images of mi and pp --> folder: mi_pp

You can accces the dataset at this URL:

https://kau365-my.sharepoint.com/:f:/g/personal/doindahj_kau_kr/ErY1Eo70vRxPhYw0ja90X48BjThy91qXOE2tyzKTgqXxQw?e=pTAOfl

preprocess>

  0. Create a text file where the paths to images are stored.(text file will be used in data augmentation code) --> file: text.ipynb
  1. crop the image for showing one footprint --> file: crop_code.ipynb
  2. data augmentation of mi and pp --> file: aug.ipynb

triplet loss>
  1. preprocess (change size to 224x224(VGG16, MobileNetv2, Exception, ResNet) or 331x331(NASNetLarge)) --> file: 1_data_preprocessing.py
  2. classify species (in this project only dog exist) --> file: 2_species_classification
  3. identification classification (bori, mi, pony, pp, wangbal) --> file: 3_individual_identification.py
  4. evaluation (tSNE, accuracy) --> file: 4_evaluation_of_3.py
     
     실제로 4과정에서는 t-SNE의 결과를 보기 위해서 구글 코랩에서 실험을 진행하였다.
     
     그 결과는 4_evaluation_of_3.ipynb 파일에서 볼 수 있다.

Run the code by using "python [file name]" command at linux system.

reference : https://github.com/jtdsouza/w251-WildTrackAI.git
