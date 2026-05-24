import scipy.io as sio
import numpy as np 
from units.ParserConfig import *
from units.unit import normalize 
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(dataset_name):
    args = parse_args()
    view_num = args.view_num
      
    if dataset_name == 'BDGP':
        '''
            This is drosophila embryos, each of which is represented by visual and textual features.
            view 1 and 2 is represented by visual and textual features.
            view_num = 2, classes = 5, samples = 2500
        '''
        view_num = 2
        mat = sio.loadmat('./data/BDGP.mat')
        X = [normalize(mat['X1'].astype(np.float32), flag='row_vector'),
             normalize(mat['X2'].astype(np.float32), flag='row_vector')]
        y = np.squeeze(mat['Y']).astype('int')
        X_list = []
        for view in range(view_num):
            X_list.append(X[view])
        if np.min(y) == 1:
            y = y - 1
        return X_list, y


    elif dataset_name == 'BBCSport':
        '''
             BBCSportL: Contains 544 samples in 5 clusters for 2 views  3183/3203, 
              '''
        X_list = []
        mat = sio.loadmat('./data/BBCSport/BBCSport')
        for i in range(view_num):
            x = mat['X'][0][i].T
            x = x.toarray()

            # print(x.shape)
            # x = normalize(mat['X'][0][i].T, flag="row_vector")
            X_list.append(normalize(x, flag='row_vector'))

        y = np.squeeze(mat['Y']).astype('int')
        if np.min(y) == 1:
            y = y - 1
        return X_list, y

    elif dataset_name == 'NGs':
        '''
            contains 500 instance in 5 clusters for 3 vies and feature dimension is [2000, 2000, 2000]
        '''
        mat = sio.loadmat('./data/NGs.mat')
        X_list = []
        for view in range(3):
            X = mat['X']
            X_list.append(normalize(X[view][0], flag='row_vector'))
        y = np.squeeze(mat['Y']).astype('int')
        if np.min(y) == 1:
            y = y - 1
        return X_list, y

    elif dataset_name == 'Hdigit':
        '''
            contains 10000 instance in 10 clusters for 2 views and feature dimension is  [784, 256]
        '''
        X_list = []
        mat = sio.loadmat('./data/Hdigit.mat')
        X = mat['data'][0]
        for view in range(view_num):
            X_list.append(normalize(X[view].T))
        y = np.squeeze(mat['truelabel'][0][0]).astype('int')
        if np.min(y) == 1:
            y -= 1
        return X_list, y
    elif dataset_name == 'cifar10':
        '''contains 50000 instance in 10 clusters for 3 views and feature dimension is  [512, 2048, 1024]'''
        X_list = []
        mat = sio.loadmat('./data/cifar10.mat')
        X = mat['data']
        for view in range(view_num):  # view_num = 3
            X_list.append(normalize(X[view][0].T, flag='row_vector'))  # sigma, row
        y = np.squeeze(mat['truelabel'][0][0]).astype('int')
        if np.min(y) == 1:
            y -= 1
        return X_list, y

    elif dataset_name == 'cora':
        '''contains 2708 instance in 7 clusters for 2 views and feature dimension is  [2708, 1433]'''
        X_list = []
        mat = sio.loadmat('./data/Cora.mat')
        X1 = mat['coracites']
        X2 = mat['coracontent']
        for i in [X1, X2]:
            X_list.append(normalize(i, 'row_vector'))
        y = np.squeeze(mat['y']).astype('int')
        if np.min(y) == 1:
            y -= 1
        return X_list, y

    elif dataset_name == 'Movies':
        '''
            contains 617 instance in 17 clusters for 2 views and feature dimension is [1878, 1398]
            It is a movie corpus extracted from IMDb. X. Xie and Y. Xiong, “Generalized multi-view learning based on generalized eigenvalues proximal support vector machines,” Expert Systems with Applications, vol. 194, p. 116491, May 2022, doi: 10.1016/j.eswa.2021.116491.
        '''
        X_list = []
        mat = sio.loadmat('./data/Movies.mat')
        X = mat['X']
        for i in range(view_num):
            X_list.append(normalize(X[i][0], flag='row_vector'))
            print(X[i][0].shape)

        y = np.squeeze(mat['y']).astype('int')
        if np.min(y) == 1:
            y -= 1

        return X_list, y
