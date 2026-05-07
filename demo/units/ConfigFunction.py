import numpy as np
 
def get_default_config(data_name):
    if data_name == 'BDGP':
        return dict(
            Autoencoder=dict(
                arch1=[1750, 1024, 1024, 1024, 128],
                arch2=[79, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
        )

