# AIcapstonedesign2021
AI capstone design 2021

The project was produced as the final project for the AI capstoneedesign 2021 class.

Participants are Lee Ho-jeong and Hwang Hye-min.

dataset
  1. Cropped Images of Original Images (for One footprint) --> folder : crop_images
  2. augmented Images of mi and pp --> folder : mi_pp

You can accces the dataset at this URL:

https://kau365-my.sharepoint.com/:f:/g/personal/doindahj_kau_kr/ErY1Eo70vRxPhYw0ja90X48BjThy91qXOE2tyzKTgqXxQw?e=pTAOfl

preprocess
  1. crop the image for showing one footprint
  2. data augmentation of mi and pp

triplet loss
  1. preporcess (change size to 224x224(VGG16, MobileNetv2, Exception, ResNet) or 331x331(NASNetLarge))
  2. classify species (in this project only dog exist)
  3. identification classification (bori, mi, pony, pp, wangbal)
  4. evaluation (tSNE, accuracy)

Run the code by using "python [file name]" command at linux system.

reference : https://github.com/jtdsouza/w251-WildTrackAI.git
