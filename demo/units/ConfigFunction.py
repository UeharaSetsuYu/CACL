def get_default_config(data_name):
    if data_name == 'BBCSport':
        return dict(
            Autoencoder=dict(
                arch1=[3183, 1024, 1024, 1024, 128],
                arch2=[3203, 1024, 1024, 1024, 128], 
                activations='relu',
                batchnorm=True,
            )
        )
'''
        config = get_default_config('')
    print(config['Autoencoder']['arch1'])
'''
