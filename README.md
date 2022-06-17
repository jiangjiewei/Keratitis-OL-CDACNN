# Keratitis-OL-CDAN
This repository contains the source code for developing an object localization combined with ensemble cost-sensitive dense attention convolutional neural network (OL-CDAN) system for the automated classification of keratitis, other cornea abnormalities, and normal cornea from slit-lamp images.  
This system provides a practical strategy for automatic diagnosis of keratitis.

# Prerequisites
* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5

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
* sikit-learn：0.23.2
* mmcv-full: 1.2.4
# Install dependencies
pip install -r requirements.txt
# Usage
* The file "object-localization" in /Keratitis-OL-CDAN is used for automatic localization for the corneal region and the conjunctival and corneal region.
* The file "CDAN" in /Keratitis-OL-CDAN is used for automatic diagnosis of keratitis.

**Please feel free to contact us for any questions or comments: Jiewei Jiang, E-mail: jiangjw924@126.com or Wei liu, E-mail: liuw_5@qq.com.**
