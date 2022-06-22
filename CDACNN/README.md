# Introduction
This repository contains the source code for developing a cost-sensitive deep attention convolutional network (CDACNN) for the automated classification of keratitis, other cornea abnormalities, and normal cornea from slit-lamp images.  
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

## Train CDACNN with conjunctival and corneal dataset on GPU
python train.py --data 'conj-c' --attention 

## Train DenseNet with conjunctival and corneal dataset on GPU
python train.py --data 'conj-c' 

## Train CDACNN with corneal dataset on GPU
python train.py --data 'c' --attention 

## Train DenseNet with corneal dataset on GPU
python train.py --data 'c'

## Train CDACNN with original dataset on GPU
python train.py --data 'ori' --attention 

## Train DenseNet with original dataset on GPU
python train.py --data 'ori'

## Evaluate CDACNN with conjunctival and corneal dataset on GPU
python test.py --data 'conj-c' --attention

## Evaluate DenseNet with conjunctival and corneal dataset on GPU
python test.py --data 'conj-c' 

## Evaluate CDACNN with corneal dataset on GPU
python test.py --data 'c' --attention

## Evaluate DenseNet with corneal dataset on GPU
python test.py --data 'c' 

## Evaluate CDACNN with original dataset on GPU
python test.py --data 'ori' --attention

## Evaluate DenseNet with original dataset on GPU
python test.py --data 'ori'
***

The representative samples for keratitis, other cornea abnormalities, and normal cornea are presented in /Keratitis-OL-CDACNN/sample.  
The representative samples of original slit-lamp images: Keratitis-OL-CDACNN/sample/Representative samples of original slit-lamp images/  
The representative samples of the conjunctival and corneal region images: Keratitis-OL-CDACNN/sample/Representative samples of the conjunctival and corneal region images/ 
The representative samples of  the corneal region images: Keratitis-OL-CDACNN/sample/Representative samples of the corneal region images/
The expected output: print the classification probabilities for keratitis, other cornea abnormalities, and normal cornea.

**Please feel free to contact us for any questions or comments: Jiewei Jiang, E-mail: jiangjw924@126.com or Wei liu, E-mail: liuw_5@qq.com.**
