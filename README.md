# Image-classification-CIFAR10

This project is the assignment part of the Deep Learning course followed 
at DTSI. Our team chose to work on the famous CIFAR10 dataset. We tried to 
develop our own alogrithm to fulfill an image classification task and compare pretrained models.

# Dataset

The dataset can be downloaded from Huggings-Face on the web page : 
https://huggingface.co/datasets/uoft-cs/cifar10. Moreover, it can be 
downloaded by following the instructions at : 
https://huggingface.co/docs/hub/datasets-usage.

A notebook explores briefly the dataset chosen and its properties (Data 
distillation.ipynb)

# Requirements and installation

The python version used is 3.10. The requirements.txt file contains all the packages we used to make the project. These packages aren't configured to handle GPU in order to ensure compatibility across platforms.

You can create a docker image of the project with the command : "docker build -t <name_container>" and by using the Dockerfile and launch it with "docker run -p 5000:5000 <name_container>". In a web adress, at the adress 0.0.0.0:5000, you will be able to test our algorithm (without transfer learning) on image of resolution 32x32.


# Models 

Models are trained using PyTorch library. We created a validation set within the given train set.
Two notebooks describe the steps we took in our exploration.
