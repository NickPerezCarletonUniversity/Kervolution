# Kervolutional Neural Networks
A Tensorflow implementation of the [Kervolutional Neural Networks (KNN)](https://arxiv.org/pdf/1904.03955.pdf) paper.

### Introduction

The KNN paper introduces an alternative operator to the usual convolution operator in CNNs, called *kernel convolution*.
The key idea is to use non-linear kernels to extract more complexe features without adding any additional parameters. According to the authors, using kernels as a source of non-linearity is supposedly more effective than using activation functions and max pooling operations.

The purpose of this repository is to further test the KNN paper's claims and also further analyze why KNNs may or may not be better than CNNs and under which circumstances.

### Changes Made
This code repository is a modification and augmentation of the repository found here: https://github.com/amalF/Kervolution. For a detailed track of all the exact changes made, go to https://github.com/NickPerezCarletonUniversity/Kervolution and click on any of the files, then the 'history' button to see a list of commits showing the exact code changes. The changes I made from the original repository are outlined below.
#### train_evaluate.py
Modified this file to enable the newly proposed methodology outlined in section 4 B of the associated KNN_Mini_Paper.pdf file.
Also enabled saving results as np arrays for creating more descriptive graphs.
#### layers.py
Added mixture of kernels implementation.
#### Create_Plots.ipynb
Created this jupyter notebook file for creating comprehensive graphs showing experiment results. Unfortunately tensorboard is limited at what it can do... (December 2020)
#### datasets.py
Modified this file to include the fashion mnist data set as well as partition datasets into k folds.
#### graphs
Added this folder showing all graphs generated by Create_Plots.ipynb.
#### final_results
Added this folder for keeping track of np arrays saved for generating graphs.
#### overfitting_logs
Added this folder for keeping track of np arrays saved for generating graphs.
#### KNN_Mini_Paper.pdf
Written report associated with this project summarizing the original KNN paper and the experimentation and analysis this repository was used for.
#### KNN_Mini_Paper.png
Png version of the above file.

### Implementation

This code was tested using *TF2.3.1* and *python 3.8.5*.

```python
pip install -r requirements.txt
```

To launch training using *LeNet5* and *MNIST* dataset as described in section 4 of the original paper :
```python
python train_evaluate.py --lr 0.003 --batch_size 50 --epochs 20 --model_name lenetknn --kernel polynomial --trainable_kernel true
```

To launch training using *LeNet5* and *MNIST* dataset as described in section 6 of the mini paper included in this repo:
```python
python train_evaluate.py --batch_size 50 --epochs 20 --model_name lenetknn --kernel polynomial --trainable_kernel true --lr_search true
```

The figure below is a png rendering of a written report summarizing the original KNN paper and the experimentation and analysis this repository was used for.  
<br />
<div align="center">
<img width="90%" src ="./KNN_Mini_Paper.png" /> 
<br />

The figures below represent the test accuracy for the first epoch.  
<br />
<div align="center">
<img width="90%" src ="./images/KernelsVsConvergence.png" /> 
<br />

For the learnable parameter cp of the polynomial kernal, the initialization of this parameter is important for faster convergence. The curve in the figure below used 0.5 as initial value.

<br />
<div align="center">
<img width="90%" src ="./images/HyperparametersVsConvergence.png" />
<br />

To test the non-linearity impact on the performance, the activations are removed and the max pooling is replaced by an average pooling. These experiments are done using a lower leraning rate (0.0001)

<br />
<div align="center">
<img width="90%" src ="./images/NonlinearityVsConvergence.png" />
<br />

### Licence
MIT - original code can be found at https://github.com/amalF/Kervolution, who also uses the same license
