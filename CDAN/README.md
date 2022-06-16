# Introduction
This repository contains the source code for developing an ensemble cost-sensitive and dense attention convolutional network (CDAN) for the automated classification of keratitis, other cornea abnormalities, and normal cornea from slit-lamp images.  
This study provides a practical strategy for automatic diagnosis of keratitis.

# Prerequisites
* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5

This repository has been tested on NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

# Installation
Other packages are as follows:
* pytorch: 1.6.0 
* wheel:  0.34.2
* yaml:   0.2.5
* scipy:  1.5.2
* joblib: 0.16.0
* opencv-python: 4.3.0.38
* scikit-image: 0.17.2
* numpy: 1.19.1
* matplotlib：3.3.1
* sikit-learn：0.23.2
# Install dependencies
pip install -r requirements.txt
# Usage
* The file "train.py" is used for our models training.
* The file "test.py"  is used for testing.

The training and testing are executed as follows:

## Train CDAN with conjunctival and corneal dataset on GPU
python train.py --data 'conj-c' --attention False

## Train ResNet50 on GPU
python keratitis_training_v1.py -a 'resnet50'

## Train Inception-v3 on GPU
python keratitis_training_v1.py -a 'inception_v3'

## Evaluate three models of DenseNet121, ResNet50, and Inception-v3 at the same time on GPU
python keratitis_testing_v1.py
***

The representative samples for keratitis, other cornea abnormalities, and normal cornea are presented in /Keratitis-Source/sample.  
The representative samples of slit-lamp images: Keratitis-Source/sample/Representative samples of slit-lamp images/  
The representative samples of smartphone images: Keratitis-Source/sample/Representative samples of smartphone images/  
The expected output: print the classification probabilities for keratitis, other cornea abnormalities, and normal cornea.

**Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Jiewei Jiang, E-mail: jiangjw924@126.com.**