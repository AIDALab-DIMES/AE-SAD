import tensorflow as tf
from tensorflow.keras import layers,losses
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import roc_auc_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import create_dataset
from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
import copy
import random




class Shallow_Autoencoder(Model):
    def __init__(self,dim,flat_dim,latent_dim):
        super(Shallow_Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(flat_dim, activation='sigmoid'),
            layers.Reshape(dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Deep_Autoencoder(Model):
    def __init__(self,dim,flat_dim,intermediate_dim,latent_dim):
        super(Deep_Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(intermediate_dim, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(intermediate_dim, activation='relu'),
            layers.Dense(flat_dim, activation='sigmoid'),
            layers.Reshape(dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Conv_Autoencoder(Model):
    def __init__(self, dim):
        super(Conv_Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=dim),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class PCA_Autoencoder(Model):
    def __init__(self,dim,flat_dim,latent_dim):
        super(PCA_Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='linear', use_bias=False),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(flat_dim, activation='linear', use_bias=False),
            layers.Reshape(dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
