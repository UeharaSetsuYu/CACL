import itertools
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import collections
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Function
from units.datasets import load_data
from units.unit import set_requires_grad, GRL_coeff, pseudo_clustering, get_supervised_metrics
from model import *
from units.loss import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED = True
import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
def setup_seed(seed): 
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)
    random.seed(seed+4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reconstruction(model, data_list, view_num, Criterion, optimizer, target_list, Flag, mask = None) :
    x_hat, *_ = model(
        data_list)  # type is list overall
    loss_list = []
    loss_1 = 0
    optimizer.zero_grad()
    X = []
    for view in range(view_num):
        X.append(x_hat[view].detach().cpu().numpy())
        loss_list.append(Criterion.forward_mse(x_hat[view], data_list[view]))
    loss = sum(loss_list)  # Reconstruction Loss only, now
    loss_1 += loss.item() / view_num
  
    loss.backward()
    optimizer.step()

    return loss_1
def UCRM(model, data_list, view_num, Criterion, optimizer):
    loss_list = []
    *_, pseudo_list, _, _= model(data_list)
    loss_3 = 0
    optimizer.zero_grad()
    for i in range(view_num):
        for j in range(i + 1, view_num):
            q_m = pseudo_list[i]
            q_n = pseudo_list[j]
            loss_align = Criterion.pseudoAlignLoss(q_m, q_n)
            # loss_align = Criterion.forward_label(q_m, q_n)
            loss_list.append(loss_align)

    loss = 0.5 * (sum(loss_list)) / view_num
    loss_3 += loss.item()


    loss.backward()
    optimizer.step()


    return loss_3


def Classification(model, data_list, view_num, Criterion, optimizer, y):
    loss_list = []
    loss_list_cla = []
    y = list(y)
    for i in range(view_num): y[i] = y[i].to(data_list[i].device).long()
    *_, pseudo_list, _, _ = model(data_list)
    loss_3 = 0
    optimizer.zero_grad()
    CE_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    # CE_loss = nn.CrossEntropyLoss()
    # CE_loss = Criterion.forward_
    for i in range(view_num):
 
        for j in range(i + 1, view_num):
            q_m = pseudo_list[i]
            q_n = pseudo_list[j]
            loss_align = Criterion.pseudoAlignLoss(q_m, q_n)

            loss_list.append(loss_align)
    loss_1 = 0.5 * (sum(loss_list)) / view_num

    for i in range(view_num):
        q_v = pseudo_list[i] 
        loss_list_cla.append(Criterion.forward_Focal_CE(q_v, y[i]))

    loss_2 = sum(loss_list_cla) / view_num

    loss = loss_1 * 0.1 + loss_2
    loss_3 += loss.item()

    loss.backward()
    optimizer.step()
    return loss_3
def RecUCRM(model, data_list, view_num, Criterion, optimizer):
    x_hat, *_ = model(
        data_list)  # type is list overall
    optimizer.zero_grad()
    loss_list = []
    loss_1 = 0
    for view in range(view_num):
        loss_list.append(Criterion.forward_mse(x_hat[view], data_list[view]))
    loss = sum(loss_list)  # Reconstruction Loss only, now
    loss_1 += loss.item() / view_num

    loss_list = []
    *_, pseudo_list, _, _ = model(data_list)
    loss_3 = 0

    for i in range(view_num):
        for j in range(i + 1, view_num):
            q_m = pseudo_list[i]
            q_n = pseudo_list[j]
            loss_align = Criterion.pseudoAlignLoss(q_m, q_n)
            # loss_align = Criterion.forward_label(q_m, q_n)
            loss_list.append(loss_align)

    loss_clu = 0.5 * (sum(loss_list)) / view_num
    loss_3 += loss_clu.item()
    loss += 0.1 * loss_clu



    loss.backward()
    optimizer.step()

    return loss_1, loss_3

def GRCAM_Bi(model, data_list, view_num, Criterion, args, device, optimizer_CRMN, optimizer_Cla, iter_idx, num_iters):


    '''Training Classifier'''
    set_requires_grad(model.classifier, True)
    set_requires_grad(model.MLP, False)
    model.classifier.train()
    model.MLP.eval()
    _, latent_share, *_, z_share = model(data_list)
    optimizer_Cla.zero_grad()
    for _ in range(args.n_critic):
        loss_list_class = []
        for view in range(view_num):
            shared_z = z_share[view].detach()

            specific_label = torch.full((shared_z.shape[0],), view, dtype=torch.long, device=device)
            specific_logit = model.classifier(shared_z)

            loss_cla = Criterion.forward_CrossEntropy(specific_logit, specific_label)
            loss_list_class.append(loss_cla)
        loss_classifier = sum(loss_list_class) / view_num

        loss_1 = loss_classifier.item()
        loss_classifier.backward()
        optimizer_Cla.step()

    '''Training CRMN'''
    optimizer_CRMN.zero_grad()
    set_requires_grad(model.classifier, False)
    set_requires_grad(model.MLP, True)
    model.classifier.eval()
    model.MLP.train()
    _, latent_share, *_, z_share = model(data_list)
    loss_list_con = []
    for view in range(view_num):
        shared_z = z_share[view]
        # lambd = 1
        beta = args.beta
        lambd = GRL_coeff(iter_idx,beta, num_iters)    # best lambda is 3
        # lambd = linear_lambda(iter_idx / float(num_iters), max_lambda=1.0, ramp_up_end=0.5)
        shared_z_rev = GRL(shared_z, lambd = lambd)
        specific_label = torch.full((shared_z.shape[0],), view, dtype=torch.long, device=device)
        shared_logit = model.classifier(shared_z_rev)

        loss_con = Criterion.forward_CrossEntropy(shared_logit, specific_label)
        loss_list_con.append(loss_con)

    loss_consistency = sum(loss_list_con) / view_num
    loss_2 = loss_consistency.item()
    loss_consistency.backward()
    optimizer_CRMN.step()

    return loss_1, loss_2

def GRCAM_Bi_woAdv(model, data_list, view_num, Criterion, optimizer_CRMN):
    '''
        *_, z_share = model(data_list) , z_share is consistency representations, type is list
    '''

    *_, z_share = model(data_list)

    '''Training CRMN'''
    optimizer_CRMN.zero_grad()
    model.MLP.train()

    loss_list_con = []


    for i in range(view_num):
        for j in range(i + 1, view_num):
            z_1 = z_share[i]
            z_2 = z_share[j]

            # MI based
            # loss_view = Criterion.forward_MI(z_1, z_2)

            # MSE based
            loss_view = Criterion.mse(z_1, z_2)

            loss_list_con.append(loss_view)






    loss_consistency = sum(loss_list_con) / view_num
    loss_1 = loss_consistency.item()
    loss_consistency.backward()
    optimizer_CRMN.step()

    return loss_1

def CRFN_Adversarial(model, data_list, view_num, Criterion, args, device, optimizer_CRFN, optimizer_Disc):
    loss_list_disc = []

    loss_1, loss_2 = 0, 0
    '''Training Discriminator'''
    set_requires_grad(model.discriminator, True)
    set_requires_grad(model.filter, False)
    model.filter.eval()
    model.discriminator.train()
    _, latent_share, latent_specific, hlz, z_con, *_, z_share = model(data_list)
    optimizer_Disc.zero_grad()

    share_label = torch.full((z_share[0].shape[0], ), 0, dtype=torch.long, device=device)
    aplha = args.epsilon
    eta = args.eta
    for view in range(view_num):
        specific_label = torch.full((z_share[0].shape[0],), view + 1, dtype=torch.long, device=device)
        specific_logit = model.discriminator(z_con[view].detach())
        shared_logit = model.discriminator(z_share[view].detach())
        loss_ce = aplha * Criterion.forward_CrossEntropy(specific_logit, specific_label) + (1 - aplha)*Criterion.forward_CrossEntropy(shared_logit, share_label)  + \
                  eta * Criterion.forward_Entropy(specific_logit)
        loss_list_disc.append(loss_ce)

    loss_disc = sum(loss_list_disc)

    loss_1 += loss_disc.item() / view_num
    loss_disc.backward()
    optimizer_Disc.step()


    '''Training Filter'''
    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.filter, True)
    model.filter.train()
    model.discriminator.eval()
    _, latent_share, latent_specific, hlz, z_con, *_, z_share = model(data_list)
    optimizer_CRFN.zero_grad()
    loss_list_crfn = []

    for view in range(view_num):
        specific_label = torch.full((z_share[0].shape[0],), view + 1, dtype=torch.long, device=device)
        specific_logit = model.discriminator(z_con[view])
        shared_logit = model.discriminator(z_share[view].detach())
        loss_filter = aplha * Criterion.forward_CrossEntropy(specific_logit, specific_label) + (
                1 - aplha) * Criterion.forward_CrossEntropy(shared_logit, share_label) + \
                  eta * Criterion.forward_Entropy(specific_logit)
        loss_list_crfn.append(loss_filter)
    loss_filter = sum(loss_list_crfn)
    loss_2 += loss_filter.item() / view_num
    loss_filter.backward()
    optimizer_CRFN.step()

    return loss_1, loss_2


def CRFN_Adversarial_NoneConsis(model, data_list, view_num, Criterion, args, device, optimizer_CRFN, optimizer_Disc):
    loss_list_disc = []

    loss_1, loss_2 = 0, 0
    '''Training Discriminator'''
    set_requires_grad(model.discriminator, True)
    set_requires_grad(model.filter, False)
    model.filter.eval()
    model.discriminator.train()
    _, latent_share, latent_specific, hlz, z_con, *_, z_share = model(data_list)
    optimizer_Disc.zero_grad()

    aplha = args.epsilon
    eta = 10
    for view in range(view_num):
        specific_label = torch.full((z_share[0].shape[0],), view , dtype=torch.long, device=device)
        specific_logit = model.discriminator(z_con[view].detach())
        shared_logit = model.discriminator(z_share[view].detach())
        loss_ce = aplha * Criterion.forward_CrossEntropy(specific_logit, specific_label) + eta * Criterion.forward_Entropy(specific_logit)
        loss_list_disc.append(loss_ce)

    loss_disc = sum(loss_list_disc)

    loss_1 += loss_disc.item() / view_num
    loss_disc.backward()
    optimizer_Disc.step()


    '''Training Filter'''
    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.filter, True)
    model.filter.train()
    model.discriminator.eval()
    _, latent_share, latent_specific, hlz, z_con, *_, z_share = model(data_list)
    optimizer_CRFN.zero_grad()
    loss_list_crfn = []

    for view in range(view_num):
        specific_label = torch.full((z_share[0].shape[0],), view , dtype=torch.long, device=device)
        specific_logit = model.discriminator(z_con[view])
        loss_filter = aplha * Criterion.forward_CrossEntropy(specific_logit, specific_label) + eta * Criterion.forward_Entropy(specific_logit)
        loss_list_crfn.append(loss_filter)
    loss_filter = sum(loss_list_crfn)
    loss_2 += loss_filter.item() / view_num
    loss_filter.backward()
    optimizer_CRFN.step()

    return loss_1, loss_2

def main_train(args, dataset_name, config, device):
    setup_seed(args.seed)
    train_loader, test_loader, n_cluster = data_preprocess(dataset_name, args)




    epochs = args.epochs
    loss_metrics = collections.defaultdict(list)
    accumulated_metrics = collections.defaultdict(list)
    bestmodel_metrics = collections.defaultdict(list)
    (accumulated_metrics['bestacc'], accumulated_metrics['bestari'], accumulated_metrics['bestnmi'], accumulated_metrics['bestsc'],
     accumulated_metrics['bestpur'], accumulated_metrics['bestfmea'])  = 0, 0, 0, 0, 0, 0
    view_num = args.view_num
    dim = [config['Autoencoder']['arch' + str(i + 1)] for i in range(view_num)]
    model = CACL(auto_dim= dim, device = device,
                  view_num = view_num, cluster_n = n_cluster, args = args)


    model = model.to(device) 
    '''Training Discriminator and Generator, respectively'''
   
    t_max_epoch = epochs
    lr = 1e-3
    optimizer_CRMN = optim.Adam([
        {'params': model.MLP.parameters()},
        {'params': model.Encoder_c.parameters()}], lr = args.lr)
    scheduler_CRMN = CosineAnnealingLR(optimizer_CRMN, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_CRFN = optim.Adam([{'params': model.filter.parameters()}], lr = args.lr)  # , {'params': model.Encoder_s.parameters()}
    scheduler_CRFN = CosineAnnealingLR(optimizer_CRFN, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_Disc = optim.Adam(model.discriminator.parameters(), lr = lr)
    scheduler_Disc = CosineAnnealingLR(optimizer_Disc, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_Rec = optim.Adam([ 
        {'params': model.Encoder_s.parameters()},
        {'params': model.Decoder.parameters()}
    ], lr=args.lr)
    optimizer_UCM = optim.Adam(model.pseudo_mlp.parameters(), lr = lr)

    optimizer_Cla = optim.Adam(model.classifier.parameters(), lr = lr)  # New setting
    scheduler_Cla = CosineAnnealingLR(optimizer_Cla, T_max = t_max_epoch, eta_min=1e-6)

    optimizer_RecUCM = optim.Adam([
        {'params': model.Encoder_s.parameters()}, {'params': model.Decoder.parameters()}, {'params': model.pseudo_mlp.parameters()}
    ], lr = args.lr)
    shecduler_RecUCM = CosineAnnealingLR(optimizer_RecUCM, T_max=t_max_epoch, eta_min=1e-6)
    #
 
    ''' Training '''
    # print("======================= Stage 1 =======================")
    pre_train = args.pre_train
    num_iters = len(train_loader) *  pre_train # BDGP datasets best iter_num = 400 on linear, but 300 on sigmoid
    iter_idx = 0

    Flag = False
    flag = True

    for epoch in range(epochs):
        if epoch == 55555:
            Flag = True
 
        model.train()
        loop = tqdm(enumerate(zip(*train_loader)), desc=f'Training Processing:  {epoch + 1} / {epochs} ',total=len(train_loader[0]), # if you want bar, you can copy total=len(train_loader[0]),
                    leave=True, dynamic_ncols=True)
        all_loss, loss_consis, loss_rec, loss_adver_disc, loss_adver_cla, loss_Clu, loss_comple = 0, 0, 0, 0, 0, 0, 0 # loss_1 is L_res

        '''Datasets Initialization'''
        for batch_idx, batch in loop:

            data_list, target_list = zip(*batch)
            data_list = list(data_list)



            for view in range(view_num):
                data_list[view] = data_list[view].to(device)


            '''Loss Function Initialization'''
            Criterion = Loss(n_cluster)

            '''Training Consistency Adversarial Module'''
            if epoch < pre_train:

                loss_1, loss_2 = GRCAM_Bi(model, data_list, view_num, Criterion, args, device, optimizer_CRMN, optimizer_Cla, iter_idx, num_iters)

                iter_idx += 1
                loss_adver_cla += loss_1
                loss_consis += loss_2

                # loss_1 = GRCAM_Bi_woAdv(model, data_list, view_num, Criterion, optimizer_CRMN)
                # loss_adver_cla += 0
                # loss_consis = 0

            else:
                set_requires_grad(model.classifier, False)
                set_requires_grad(model.MLP, False)
                model.classifier.eval()
                # set_requires_grad(model.Encoder_c, False)
                model.MLP.eval()
                #
                #
                loss_1, loss_2 = CRFN_Adversarial_NoneConsis(model, data_list, view_num, Criterion, args, device, optimizer_CRFN, optimizer_Disc)
     
                loss_adver_disc += loss_1
                loss_comple += loss_2
 

                loss_rec += reconstruction(model, data_list, view_num, Criterion, optimizer_Rec, target_list, Flag)
                if args.model == 'Clustering':
                    loss_Clu += UCRM(model, data_list, view_num, Criterion, optimizer_UCM)
                elif args.model == 'Classification':
                    loss_Clu += Classification(model, data_list, view_num, Criterion, optimizer_UCM, target_list)

            all_loss += loss_consis + loss_adver_cla + loss_adver_disc + loss_rec + loss_Clu + loss_adver_cla + loss_comple
            loop.set_postfix(Recon_Loss=f"{loss_rec: .6f}",
                             All_Loss=f'{all_loss: .6f}',
                             Con_Loss = f"{loss_consis: .6f}",
                             Comple_Loss = f"{loss_comple: .6f}",
                             zDiscri_Loss = f"{loss_adver_disc: .6f}",
                             zClassifer_Loss = f"{loss_adver_cla: .6f}",
                             Pseudo_Loss = f"{loss_Clu: .6f}")
            '''Stage one Loss'''
            loss_metrics['Consistency_Loss'].append(loss_consis)    # Adversarial Loss by ConRE
            loss_metrics['Classifier_Loss'].append(loss_adver_cla)  # Adversarial Loss by Discriminator
            '''Stage two Loss'''
            # loss_metrics['Discriminator'].append(loss_adver_disc)   # Classifier Loss
            loss_metrics['Complementarity'].append(loss_adver_disc + loss_comple + loss_Clu + loss_rec)
            '''All Loss'''
            loss_metrics['all_loss'].append(all_loss)


        scheduler_Disc.step()
        scheduler_CRFN.step()
        scheduler_CRMN.step()
        scheduler_Cla.step()
        shecduler_RecUCM.step()

        if (epoch + 1) % 10 == 0 and epoch > args.pre_train:
            Evaluation(model, test_loader, device, accumulated_metrics, view_num, bestmodel_metrics, epoch)

            if (epoch + 1) % 10 == 0:
                if args.model == 'Classification':
                    Evaluation_classifcation(model, test_loader, device, accumulated_metrics, view_num,
                                             bestmodel_metrics, epoch)
                    print(f'Best clustering performance: acc: {accumulated_metrics["acc"][-1]}, '
                          f'precision: {accumulated_metrics["precision"][-1]:.4f}, f_mea: {accumulated_metrics["f_mea"][-1]:.4f}')
                elif args.model == 'Clustering':
                    Evaluation(model, test_loader, device, accumulated_metrics, view_num, bestmodel_metrics, epoch)
                    print(f'Best clustering performance: acc: {accumulated_metrics["acc"][-1]}, '
                          f'ari: {accumulated_metrics["ari"][-1]}, nmi: {accumulated_metrics["nmi"][-1]}, '
                          f'pur: {accumulated_metrics["pur"][-1]:.4f}, f_mea: {accumulated_metrics["f_mea"][-1]:.4f}')

    if args.model == 'Classification':
        print(f'Best clustering performance: acc: {accumulated_metrics["bestacc"]}, ' 
              f'precision: {accumulated_metrics["best_precision"]}, f_mea: {accumulated_metrics["bestfmea"]}')
    elif args.model == 'Clustering':
        print(f'Best clustering performance: acc: {accumulated_metrics["bestacc"]}, '
              f'ari: {accumulated_metrics["bestari"]}, nmi: {accumulated_metrics["bestnmi"]}, '
              f'pur: {accumulated_metrics["bestpur"]:.4f}')

   
    return accumulated_metrics, loss_metrics

def Evaluation_classifcation(model, test_loader, device, accumulated_metrics, view_num, bestmodel_metrics, epoch):

    all_labels_true, all_labels_pred = [], []
    z_feature, feature, origin_data = [], [], []
    with torch.no_grad():  # if you want bar, you can copy total=len(train_loader[0]) in loop,
        model.eval()
        data = enumerate(zip(*test_loader))
        for batch_idx, batch in data:
            data_list, target_list = zip(*batch)
            data_list = list(data_list)
            for i in range(view_num): data_list[i] = data_list[i].to(device)
            _, latent_share, latent_specific, hlz, z_con, specific_prob, share_prob, pseudo_list, _, z_shared = model(data_list)

            '''Pseudo-Clustering Result'''
            p = torch.stack(pseudo_list)
            max_probs = torch.mean(p, dim=0)
            final_p, final_c = torch.max(max_probs, dim = 1)
            y = final_c.detach().cpu().numpy().ravel().astype(int)
            label = target_list[0].numpy().ravel()

            '''Clustering Result Recording'''
            all_labels_true.extend(label)
            all_labels_pred.extend(y) 
            
        acc, pre, f1 = get_supervised_metrics(all_labels_true, all_labels_pred)
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['precision'].append(pre)
        accumulated_metrics['f_mea'].append(f1)

        if epoch >= 0:
                if (accumulated_metrics['bestacc'] < acc):
                    accumulated_metrics['bestacc'] = acc

                    accumulated_metrics['best_precision'] = pre

                    accumulated_metrics['bestfmea'] = f1
                    bestModel = model.state_dict()

                    bestmodel_metrics['bestModel'] = bestModel
                    
def Evaluation(model, test_loader, device, accumulated_metrics, view_num, bestmodel_metrics, epoch):

    all_labels_true, all_labels_pred = [], []
    with torch.no_grad():  # if you want bar, you can copy total=len(train_loader[0]) in loop,
        model.eval()
        data = enumerate(zip(*test_loader))
        im = []
        for batch_idx, batch in data:
            data_list, target_list = zip(*batch)
            data_list = list(data_list)
            for i in range(view_num): data_list[i] = data_list[i].to(device)
            _, latent_share, latent_specific, hlz, z_con, specific_prob, share_prob, pseudo_list, _, z_shared = model(data_list) 

            '''Pseudo-Clustering Result'''
            p = torch.stack(pseudo_list)
            max_probs = torch.mean(p, dim=0)
            final_p, final_c = torch.max(max_probs, dim = 1)
            y = final_c.detach().cpu().numpy().ravel().astype(int)
            label = target_list[0].numpy().ravel()
            '''Clustering Result Recording'''
            all_labels_true.extend(label)
            all_labels_pred.extend(y)

        pseudo_clustering([], all_labels_pred, all_labels_true, accumulated_metrics['acc'],
                          accumulated_metrics['ari'],
                          accumulated_metrics['nmi'],
                          accumulated_metrics['SC'], accumulated_metrics['pur'], accumulated_metrics['f_mea'])
        im = np.mean(im)
        accumulated_metrics['MMD_Comple'].append(im)
        if epoch >= 0:
                if (accumulated_metrics['bestacc'] < accumulated_metrics['acc'][-1]):
                    accumulated_metrics['bestacc'] = accumulated_metrics['acc'][-1]
                    accumulated_metrics['bestari'] = accumulated_metrics['ari'][-1]
                    accumulated_metrics['bestnmi'] = accumulated_metrics['nmi'][-1]
                    accumulated_metrics['bestsc'] = accumulated_metrics['SC'][-1]
                    accumulated_metrics['bestpur'] = accumulated_metrics['pur'][-1]
                    accumulated_metrics['bestfmea'] = accumulated_metrics['f_mea'][-1]
                    bestModel = model.state_dict()

                    bestmodel_metrics['bestModel'] = bestModel



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def GRL(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def data_preprocess(dataset_name, args) ->list:
    '''
        :param dataset_name: Training dataset name
        :param args: args object
        :return: *A list, called train_loader and test_loader, include each view *
    '''
    data_train, y_list = load_data(dataset_name)
    # mask = (np.random.rand(args.batch_size, 128) > 0.7).astype(np.float32)
    np.random.seed(args.seed)
    train_loader, test_loader = [], []
    index_list = np.arange(len(data_train[0]))
    index_train = np.random.choice(index_list, size=int(len(index_list) * args.train_rate), replace=False)

    index_test = np.setdiff1d(index_list, index_train)  # The shuffle operation on the next line won't shuffle the data; that will happen later.
    np.random.shuffle(index_test)
    view_num = args.view_num
    n_cluster = len(np.unique(y_list))
    for i in range(view_num):
        x_train, y_train = data_train[i][index_train] , y_list[index_train]

        x_test, y_test = data_train[i][index_test], y_list[index_test]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.int32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.int32)
        dataset_train = TensorDataset(x_train_tensor, y_train_tensor)
        dataset_test = TensorDataset(x_test_tensor, y_test_tensor)
         # The data has already been shuffled and does not need to be shuffled again. (shuffle = False)
        data_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, drop_last=True)   
        data_test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
        train_loader.append(data_train_loader)
        test_loader.append(data_test_loader)


    return train_loader, test_loader, n_cluster









