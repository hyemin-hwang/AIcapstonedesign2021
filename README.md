# AIcapstonedesign2021
AI capstone design 2021
The project was produced as the final project for the AI capstoneedesign 2021 class.

Participants are Lee Ho-jeong and Hwang Hye-min.

dataset
1. Original Images(multiple footprints)
2. Cropped Images
3. augmented Images

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
