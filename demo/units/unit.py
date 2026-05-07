import numpy as np
import sklearn.metrics as metrics
from munkres import Munkres
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, f1_score, precision_score, accuracy_score
from scipy.optimize import linear_sum_assignment
 
def normalize(x, flag='default', epsilon=1e-8):
    if flag == 'default':  
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + epsilon)
    elif flag == 'col_vector':   
        x = (x - np.min(x, axis=0, keepdims=True)) / \
            (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True) + epsilon)
    elif flag == 'row_vector':   
        x = (x - np.min(x, axis=1, keepdims=True)) / \
            (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) + epsilon)
    elif flag == 'sigma':
        x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + epsilon)
    return x




class y_pre_align():
    '''
        define:
            y = y_pre_align(y_pre, y_true, n_cluster=n)
            print(y.adjust_label())
    '''
    def __init__(self, y_pre, y_true, n_cluster):
        self.y_pre = y_pre
        self.y_true = y_true
        self.n_cluster = n_cluster

    def calculate_cost_matrix(self, C, n_clusters):
        cost_matrix = np.zeros((n_clusters, n_clusters))

        # cost_matrix[i,j] will be the cost of assigning cluster i to label j
        for j in range(n_clusters):
            s = np.sum(C[:, j])  # number of examples in cluster i  被预测为j类的个数
            for i in range(n_clusters):
                t = C[i, j]   
                cost_matrix[j, i] = s - t   
        return cost_matrix   

    def get_cluster_labels_from_indices(self, indices):
        n_clusters = len(indices)
        clusterLabels = np.zeros(n_clusters)
        for i in range(n_clusters):
            clusterLabels[i] = indices[i][1]
        return clusterLabels

    def get_y_preds(self, y_true, cluster_assignments, n_clusters):  # get true labels function
 
        all_labels = list(range(n_clusters))
        confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=all_labels)    
        cost_matrix = self.calculate_cost_matrix(confusion_matrix, n_clusters)
        indices = Munkres().compute(cost_matrix)    
        kmeans_to_true_cluster_labels = self.get_cluster_labels_from_indices(indices)  
        if np.min(cluster_assignments) != 0:
            cluster_assignments = cluster_assignments - np.min(cluster_assignments)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignments]

        return y_pred

    def adjust_label(self):
        return self.get_y_preds(self.y_true, self.y_pre, self.n_cluster)






def get_supervised_metrics(all_labels_true, all_labels_pred): 
    y_true = np.array(all_labels_true)
    y_pred = np.array(all_labels_pred) 
    acc = accuracy_score(y_true, y_pred) 
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) 
    f1 = f1_score(y_true, y_pred, average='macro') 
    return round(acc, 4), round(precision, 4), round(f1, 4)


def pseudo_clustering(x, y_pred, labels, acc, ari, nmi, precision, pur, f_mea):
 
    labels = np.array(labels)
    y_pred = np.array(y_pred)
    # ---------- ACC ----------
    D = max(y_pred.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(labels)):
        w[y_pred[i], labels[i]] += 1
    ind_row, ind_col = linear_sum_assignment(w.max() - w)
    ACC = sum(w[i, j] for i, j in zip(ind_row, ind_col)) / len(labels)

    # ---------- NMI ----------
    NMI = normalized_mutual_info_score(labels, y_pred)

    # ---------- ARI ----------
    ARI = adjusted_rand_score(labels, y_pred)

    # ---------- PUR ----------
    total = 0
    for cluster in np.unique(y_pred):
        idx = np.where(y_pred == cluster)[0]
        true_labels = labels[idx]
        if len(true_labels) > 0:
            total += np.bincount(true_labels).max()
    PUR = total / len(labels)

    # ---------- SC --------------
    # SC = silhouette_score(x, y_pred)
    SC = 0
    #----------- Precision ------------
    preci = precision_score(labels, y_pred, average='macro', zero_division=0)
    # ---------- F_mea ------------
    f_1 = f1_score(labels, y_pred, average='macro')   

    acc.append(round(ACC, 4))
    ari.append(round(ARI, 4))
    nmi.append(round(NMI, 4))
    pur.append(round(PUR, 4))
    precision.append(round(preci, 4))
    f_mea.append(round(f_1, 4))
 
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def GRL_coeff(epoch, beta, n):
    coeff = 2 / (1 + np.exp(-beta * epoch / n)) - 1
    return coeff
 
