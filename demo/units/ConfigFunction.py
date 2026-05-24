import scipy.io as sio
import numpy as np 
from units.ParserConfig import *
from units.unit import normalize 
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(dataset_name):
    args = parse_args()
    view_num = args.view_num
    X_list = []
    label = 0
    if dataset_name == 'BDGP':
        '''
            This is drosophila embryos, each of which is represented by visual and textual features.
            view 1 and 2 is represented by visual and textual features.
            view_num = 2, classes = 5, samples = 2500
        '''
        mat = sio.loadmat('D:\Data_Mining\Code\Datasets\BDGP\BDGP.mat')
        X = [normalize(mat['X1'].astype(np.float32), flag='row_vector'),
             normalize(mat['X2'].astype(np.float32), flag='row_vector')]
        y = np.squeeze(mat['Y']).astype('int')
        
        for view in range(2):
            X_list.append(X[view])
        if np.min(y) == 1:
            y = y - 1
            
        label = y

    elif dataset_name == 'NGs':
        '''
            contains 500 instance in 5 clusters for 3 vies and feature dimension is [2000, 2000, 2000]
        '''
        mat = sio.loadmat('./data/NGs.mat') 
        for view in range(3):
            X = mat['X']
            X_list.append(normalize(X[view][0], flag='row_vector'))
        y = np.squeeze(mat['Y']).astype('int')
        if np.min(y) == 1:
            y = y - 1
        label = y

    elif dataset_name == 'BBCSport':
        '''
             BBCSportL: Contains 544 samples in 5 clusters for 2 views  3183/3203, 
              ''' 
        mat = sio.loadmat('./data/BBCSport/BBCSport')
        for i in range(view_num):
            x = mat['X'][0][i].T
            x = x.toarray() 
            X_list.append(normalize(x, flag='row_vector'))

        y = np.squeeze(mat['Y']).astype('int')
        if np.min(y) == 1:
            y = y - 1
        label = y


return X_list, label 
