# Welcome to this Azure Machine Learning Workshop!

In this tutorial, you will learn how to train a Pytorch image classification model using transfer learning with the Azure Machine Learning service. The Azure Machine Learning python SDK's [PyTorch estimator](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-pytorch) enables you to easily submit PyTorch training jobs for both single-node and distributed runs on Azure compute. The model is trained to classify dog breeds using the [Stanford Dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and it is based on a pretrained ResNet18 model. This ResNet18 model has been built using images and annotation from ImageNet. The Stanford Dog dataset contains 120 classes (i.e. dog breeds), however, for most of the tutorial, we will only use a subset of this dataset which includes only 10 dog breeds.

You can view the subset of the data used [here](https://github.com/heatherbshapiro/pycon-canada/tree/master/breeds-10). 

Please refer to the [dog-breed-classifier.ipynb](dog-breed-classifier.ipynb) notebook for instructions.

![Chihuahua with hat](https://raw.githubusercontent.com/heatherbshapiro/pycon-canada/master/breeds-10/train/n02085620-Chihuahua/n02085620_11258.jpg)

