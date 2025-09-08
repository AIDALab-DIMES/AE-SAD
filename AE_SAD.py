from tensorflow.keras import layers,losses
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import AE_architectures

def launch(x_training,y_training,x_test,AE_type=None,latent_dim=32,Lambda=None,EP=1000,batch_size=32,intermediate_dim=128):
    dim = x_training[0].shape
    flat_dim = x_training[0].flatten().shape[0]


    if AE_type=='shallow':
        autoencoder = AE_architectures.Shallow_Autoencoder(dim,flat_dim,latent_dim)
    if AE_type=='deep':
        autoencoder = AE_architectures.Deep_Autoencoder(dim,flat_dim,intermediate_dim,latent_dim)
    if AE_type=='conv':
        autoencoder = AE_architectures.Conv_Autoencoder(dim)
    if AE_type=='pca':
        autoencoder = AE_architectures.PCA_Autoencoder(dim,flat_dim,latent_dim)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    if Lambda is None:
        Lambda = x_training.shape[0]/np.sum(y_training)
    weights = np.ones(y_training.shape[0])
    weights[np.where(y_training == 1)] = Lambda
    x_target = copy.copy(x_training)
    x_target[np.where(y_training == 1)] = 1 - x_target[np.where(y_training == 1)]
    autoencoder.fit(x_training, x_target,
                    epochs=EP,
                    shuffle=True, batch_size=batch_size, sample_weight=weights,verbose=0)

    ### -TEST- ###
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    rec_err = np.linalg.norm(x_test.reshape(x_test.shape[0], flat_dim) - decoded_imgs.reshape(x_test.shape[0], flat_dim), axis=(1)) ** 2

    return rec_err

