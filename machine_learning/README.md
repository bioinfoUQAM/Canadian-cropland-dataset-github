# Machine Learning

This sub-repository contains instructions for running the machine learning experiments. Model architectures are contained in the "models" folder.
To train each network, create and activate a python environment with the required libraries and enter:

``` 
# create conda environment with dependencies
conda env create --name envname python=3.8
conda activate envname 
```
Then, cd to the directory containing the model and run: 

```
python 3DCNN.py
```

Before training the networks, modify the desired image set in the model script by changing the training, validation and testing directories:

``` python
# define the train, validation and test directories and get the image paths (in this case, we are using the RGB data)
train_directory = "2019/RGB/training"
val_directory = "2019/RGB/validation"
test_directory = "2019/RGB/test"
```


## Models 

### Dynamic Image Classification 

3 Dimensional Convolutional Networks (3DCNN.py) 

Reference: *Uniformizing techniques to process CT scans with 3D CNNs for tuberculosis prediction.*
Zunair, H., Rahman, A., Mohammed, N., & Cohen, J. P. (2020, October). Uniformizing techniques to process CT scans with 3D CNNs for tuberculosis prediction.
In International Workshop on PRedictive Intelligence In MEdicine (pp. 156-168). Springer, Cham.

[https://arxiv.org/abs/2007.13224]
