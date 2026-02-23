# Tensorflow implementation of AE-SAD
This repository provides a Tensorflow implementation of the AE-SAD method for (semi-)supervised anomaly detection.

For more details, refer to:


[*Reconstruction Error-based Anomaly Detection with Few Outlying Examples*.](https://www.sciencedirect.com/science/article/pii/S0925231226003991) Fabrizio Angiulli, Fabio Fassetti, Luca Ferragina. *Neurocomputing* (2026).
## Citation and Contact
If you use this code for your work, please cite the paper as:
```
@article{aesad,
title = {Reconstruction error-based anomaly detection with few outlying examples},
journal = {Neurocomputing},
volume = {675},
pages = {133002},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.133002},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226003991},
author = {Fabrizio Angiulli and Fabio Fassetti and Luca Ferragina}
}
```
If you would like to get in touch, you can write at luca.ferragina@unical.it
## Abstract
Reconstruction error-based neural architectures represent a classical deep learning approach to anomaly detection which has shown great performances.
It consists in training an AutoEncoder to reconstruct a set of examples deemed to represent the normality and then to point out as anomalies those data that show a sufficiently large reconstruction error. Unfortunately, after training these architectures often become able to well reconstruct also the anomalies in the data.    
This phenomenon is much more evident when there are anomalies in the training set. In particular when these anomalies are labeled, a setting often called semi-supervised, the best way to train an AutoEncoder is to ignore the anomalies and minimize the reconstruction error only on normal data.
The goal of this work is to investigate approaches to allow reconstruction error-based architectures to instruct the model to put known anomalies outside of the domain description of the normal data. Specifically, our strategy exploits a limited number of anomalous examples to increase the contrast between the reconstruction error associated with normal examples and those associated with both known and unknown anomalies, thus enhancing anomaly detection performances.
The experiments show that this new procedure achieves better performances than the standard autoencoder approach and the main deep learning techniques for both unsupervised and semi-supervised anomaly detection. Moreover it shows better generalization on anomalies generated with a distribution different from the one of the anomalies in the training set and robustness to normal data pollution.

## Method
Let $X=\left\\{\mathbf{x}_1,\dots,\mathbf{x}_n\right\\}$ be the training set and for each $i \in [1\dots n]$, let $y_i\in\\{0,1\\}$ be the label of $\mathbf{x}_i$, with $y_i=0$ if $\mathbf{x}_i$ belongs to the normal class and $y_i=1$ if $\mathbf{x}$ is anomalous; w.l.o.g. we assume that $X \subseteq [0,1]^d$ which is always possible to obtain by normalizing the data.
The idea of the method is to train an Autoencoder with the loss
$$\mathcal{L}_F(\mathbf{x})=\left(1-y\right)\cdot\left|\left|\mathbf{x}-\hat{\mathbf{x}}\right|\right|^2+\lambda\cdot y\cdot\left|\left|F\left(\mathbf{x}\right)-\hat{\mathbf{x}}\right|\right|^2$$
where $F:[0,1]^d\rightarrow[0,1]^d$, and $\lambda$ is an hyperparameter that controls the weight of the anomalies, in relation to the normal items, during the training.

When $\mathbf{x}$ is a normal item the contribution it brings to the loss is $\left|\left|\mathbf{x}-\hat{\mathbf{x}}\right|\right|^2$ which means that the reconstruction $\hat{\mathbf{x}}$ is forced to be similar to $\mathbf{x}$ as in the standard approach. Conversely, if $\mathbf{x}$ is an anomaly, the contribution brought to the loss is $\left|\left|F\left(\mathbf{x}\right)-\hat{\mathbf{x}}\right|\right|^2$ which means that in this case $\hat{\mathbf{x}}$ is forced to be similar to $F(\mathbf{x})$. 
Hence, the idea is that, during the training process, normal data $\mathbf{x}$ are likely to be mapped to $\mathbf{\hat{x}}$ which is as similar as possible to $\mathbf{x}$ and anomalous data $\mathbf{x}$ are likely to be mapped to $F(\mathbf{x})$ which is substantially different from $\mathbf{x}$.

## Code
The code is composed by two main files:

- AE_architectures.py, that contains all the different types of Autoencoders implemented.
- AE_SAD.py, that contains the function that is used to launch the training of our method and that outputs the score computed on a training set. The function and its parameters are
  ```
  launch(x_training,y_training,x_test,AE_type=None,latent_dim=32,Lambda=None,EP=1000,batch_size=32,intermediate_dim=128)
  ```
  - ```x_training``` is the training set $X$. It is crucial that $X\subseteq[0,1]^d$, so make sure to appropriately scale your training set before passing it to the function.
  - ```y_training``` corresponds to the label of the training set $y$. It is a boolean vector, $1$ stands for anomaly and $0$ for normal.
  - ```x_test``` is the test set, the function will output a vector that contains its anomaly score. As the training set it has to be scaled to the range $[0,1]$.
  - ```AE_type``` is a string that is used to select the type of Autoencoder to employ. It can be equal to:
    - ```'shallow'``` to select a shallow dense autoencoder with an input layer, a latent space of dimension ```latent_dim``` and an output layer;
    - ```'deep'``` to select a deep dense autoencoder with an input layer, an intermediate layer of dimension ```intermediate_dim```, a latent space of dimension ```latent_dim```, another intermediate layer of dimension ```intermediate_dim``` and an output layer;
    - ```'conv'``` to select a convolutional autoencoder with an input layer, two convolutional encoding layers and two deconvolutional decoding layers;
    - ```'pca'``` to select a PCA-like autoencoder in which all the activation functions are linear, with an input layer, a latent space of dimension ```latent_dim``` and an output layer.
  - ```Lambda``` is the parameter $\lambda$, if ```None``` it is set equal to the size of the training set divided by the number of anomalies in it as adviced in the paper.
  - ```EP``` and ```batch_size``` are the epochs and the size of the bathes used in the training.

### Note
I hope you will find the code easy to use and understand. I tried to keep it as simple as possible, the only thig about it that may results a bit tricky is that the novel loss is not actually implemented, but it is simulated in the function ```launch``` with the following three operations.
- The training set ```x_training``` is duplicated and in its copy ```x_target``` is applied the function $F$ to each anomaly. As suggested by the name, ```x_target``` represents the target of the Autoencoder reconstructions.
- The Autoencoder is compiled with the standard ```MeanSquaredError``` loss and is trained with the usual ```fit```, except for the fact that the fitting is called as ```autoencoder.fit(x_training, x_target)```, which means that we want to minimize the error between the training set and the target set.
- As for the parameter ```Lambda```, we build a vector (called ```weights```), that have $1$ in a component that corresponds to a normal item and $\lambda$ in a component that corresponds to an anomaly. This vector is passed in the fitting function as ```sample_weight=weights```.
In this way the weight of each anomaly, whose reconstruction target is $F(\mathbf{x})$, is $\lambda$ and the weight of each normal item (whose reconstruction target is just $\mathbf{x}$) is $1$, just like expected.

Currently I have only implemented the function $F(\mathbf{x})=\mathbf{1}-\mathbf{x}$. I will expand this code in order to enable the definition of any function by the user.





