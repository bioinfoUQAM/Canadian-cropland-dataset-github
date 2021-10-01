# Machine Learning
This sub-repository contains instructions for running the machine learning experiments. Model architectures are contained in the "models" folder. Before training the networks, select the desired image set by changing the 
training, validation and testing directories. 

Ex: 

# define the train, validation and test directories and get the image paths
train_directory = "2019/RGB/training"
val_directory = "2019/RGB/validation"
test_directory = "2019/RGB/test"

To train each network, simply activate a python environment with the required libraries and enter:

python "model".py

Models 

Dynamic Image Classification 

3 Dimensional Convolutional Networks (3DCNN.py) 

# Reference: *Uniformizing techniques to process CT scans with 3D CNNs for tuberculosis prediction.*
Zunair, H., Rahman, A., Mohammed, N., & Cohen, J. P. (2020, October). Uniformizing techniques to process CT scans with 3D CNNs for tuberculosis prediction.
In International Workshop on PRedictive Intelligence In MEdicine (pp. 156-168). Springer, Cham.

# [https://arxiv.org/abs/2007.13224]
