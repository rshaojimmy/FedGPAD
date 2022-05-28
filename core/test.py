import os
import os.path as osp

from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir
from misc import evaluate
from torch.nn import DataParallel
import numpy as np
import h5py
import torch.nn.functional as F


def Test(args, FeatExt, Clsfier, data_loader_test_list, data_loader_train_list, savefilename):

    tgtdatasetlistname = '-'.join(args.datasetlist_test)

    datasettestname = osp.join(savefilename+'-TO-'+tgtdatasetlistname)
    print('datasettestname: {}'.format(datasettestname))


    savepath = os.path.join(args.results_path, 'testscore', datasettestname)
    mkdir(savepath)  

    FeatExt.eval()
    Clsfier.eval()
    # FeatExt = DataParallel(FeatExt)  
    # Clsfier = DataParallel(Clsfier)  

    ####################
    # Test Scores #
    ####################      
    datalenlist = []
    for i in range(len(data_loader_test_list)):
        datalen =  len(data_loader_test_list[i])
        datalenlist.append(datalen)
    alldatalen = sum(datalenlist)
           
    test_score_list = []
    test_label_list = []

    idx = 0

    for i in range(len(data_loader_test_list)):

        for (images, _, labels) in data_loader_test_list[i]:

            images = images.cuda()
            
            _,feat = FeatExt(images)
            label_pred = Clsfier(feat)

            score = torch.sigmoid(label_pred).cpu().detach().numpy()
            labels = labels.numpy()

            test_score_list.extend(score.squeeze().reshape(-1).tolist())
            test_label_list.extend(labels.reshape(-1).tolist())

            print('SampleNum:{} in total:{}, test_scores:{}'.format(idx, alldatalen, score.squeeze()))

            idx+=1


    with h5py.File(os.path.join(savepath, 'Test_results_'+args.snapshotnum+'.h5'), 'w') as hf:
        hf.create_dataset('test_scores', data=test_score_list)
        hf.create_dataset('test_labels', data=test_label_list)


    ####################
    # Dev Scores #
    ####################

    datalenlist = []
    for i in range(len(data_loader_train_list)):
        datalen =  len(data_loader_train_list[i])
        datalenlist.append(datalen)
    alldatalen = sum(datalenlist)


    devp_score_list = []
    devp_label_list = []

    idx = 0

    for i in range(len(data_loader_train_list)):

        for (catimages, _, labels, _) in data_loader_train_list[i]:

            images = catimages.cuda()

            _,feat = FeatExt(images)
            label_pred = Clsfier(feat)

            score = torch.sigmoid(label_pred).cpu().detach().numpy()
            labels = labels.numpy()

            devp_score_list.extend(score.squeeze().reshape(-1).tolist())
            devp_label_list.extend(labels.reshape(-1).tolist())

            print('SampleNum:{} in total:{}, devp_scores:{}'.format(idx, alldatalen, score.squeeze()))

            idx+=1     


    with h5py.File(os.path.join(savepath, 'Devp_results_'+args.snapshotnum+'.h5'), 'w') as hf:
        hf.create_dataset('devp_scores', data=devp_score_list)
        hf.create_dataset('devp_labels', data=devp_label_list)

   




