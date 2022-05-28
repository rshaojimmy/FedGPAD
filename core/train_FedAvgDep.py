import itertools
import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir
from misc import evaluate
from torch.nn import DataParallel
import random
import torch.autograd as autograd
from copy import deepcopy
from itertools import permutations, combinations
from models import Update_depaux
from models.Fed import FedAvg
import copy


def Train_FedAvgDep(args, FeatExt, Clsfier, DepthEst, Decoder,
        dataset_real_list, dataset_fake_list,
        summary_writer, Saver, savefilename):

    datalenlist = []
    for i in range(len(dataset_real_list)):
        datalen =  len(dataset_real_list[i])
        datalenlist.append(datalen)
        datalen =  len(dataset_fake_list[i])
        datalenlist.append(datalen)        
    iternum = max(datalenlist)      
    print('iternum={}'.format(iternum))
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExt_glob = FeatExt
    Clsfier_glob = Clsfier
    DepthEst_glob = DepthEst

    # copy weights
    w_FeatExt_glob = FeatExt_glob.state_dict()
    w_Clsfier_glob = Clsfier_glob.state_dict()
    w_DepthEst_glob = DepthEst_glob.state_dict()

    FeatExt_specific_locals = [copy.deepcopy(FeatExt) for idx in range(len(dataset_real_list))]
    Decoder_locals = [copy.deepcopy(Decoder) for idx in range(len(dataset_real_list))]

    global_step = 0
    for epoch in range(args.epochs):
        
        data_real_list = []
        data_fake_list = []
        for i in range(len(dataset_real_list)):
            data_real_list.append(get_inf_iterator(dataset_real_list[i]))
            data_fake_list.append(get_inf_iterator(dataset_fake_list[i]))

        for cur_step in range(iternum):
            w_FeatExt_locals, w_Clsfier_locals, w_DepthEst_locals, loss_locals = [], [], [], []

            #============ one batch extraction ============#
            for idx in range(len(dataset_real_list)):
                if cur_step % args.log_step == 0:
                    print('Current Local Client {}: {}/{}'.format(args.datasetlist_train[idx], int(idx+1), len(dataset_real_list)))

                imgreal, depreal, labreal, imgreal_resize = next(data_real_list[idx])
                imgfake, depfake, labfake, imgfake_resize = next(data_fake_list[idx])

                catimgone = torch.cat([imgreal,imgfake],0).cuda()
                catimgone_resize = torch.cat([imgreal_resize,imgfake_resize],0).cuda()
                catdepthone  = torch.cat([depreal,depfake],0).cuda()
                catlabone = torch.cat([labreal,labfake],0).float().cuda() 
                
                local = Update_depaux.LocalUpdate(args=args, epoch = epoch, step=cur_step, 
                                                images=catimgone, depths=catdepthone, labels=catlabone, images_resize = catimgone_resize)
                w_FeatExt, w_FeatExt_specific, w_Clsfier, w_DepthEst, w_Decoder, loss = local.train(FeatExt=copy.deepcopy(FeatExt_glob).cuda(), 
                                                                                    FeatExt_specific=copy.deepcopy(FeatExt_specific_locals[idx]).cuda(), 
                                                                                    Clsfier=copy.deepcopy(Clsfier_glob).cuda(), 
                                                                                    DepthEst=copy.deepcopy(DepthEst_glob).cuda(),
                                                                                    Decoder=copy.deepcopy(Decoder_locals[idx]).cuda()
                                                                                    )

                w_FeatExt_locals.append(copy.deepcopy(w_FeatExt))
                w_Clsfier_locals.append(copy.deepcopy(w_Clsfier))
                w_DepthEst_locals.append(copy.deepcopy(w_DepthEst))      
                loss_locals.append(loss)

                FeatExt_specific_locals[idx].load_state_dict(w_FeatExt_specific)
                Decoder_locals[idx].load_state_dict(w_Decoder)

            # update global weights
            w_FeatExt_glob = FedAvg(w_FeatExt_locals)
            w_Clsfier_glob = FedAvg(w_Clsfier_locals)
            w_DepthEst_glob = FedAvg(w_DepthEst_locals)

            # copy weight to net_glob
            FeatExt_glob.load_state_dict(w_FeatExt_glob)
            Clsfier_glob.load_state_dict(w_Clsfier_glob)
            DepthEst_glob.load_state_dict(w_DepthEst_glob)
            # print loss
            if cur_step % args.log_step == 0:
                loss_avg = sum(loss_locals) / len(loss_locals)
                print('Epoch {:3d}, Step: {}, Average loss {:.3f}'.format(epoch, cur_step, loss_avg))



            #============ tensorboard the log info ============#
            info = {
                'loss_avg': loss_avg,                                                                                                                                                                                                                                                                 
                    }           
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step) 
            global_step+=1

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)     
            mkdir(model_save_path) 

            torch.save(FeatExt_glob.state_dict(), os.path.join(model_save_path,
                "FeatExt-{}.pt".format(epoch+1)))
            torch.save(Clsfier_glob.state_dict(), os.path.join(model_save_path,
                "Clsfier-{}.pt".format(epoch+1)))
            torch.save(DepthEst_glob.state_dict(), os.path.join(model_save_path,
                "DepthEst-{}.pt".format(epoch+1)))

            for i in range(len(FeatExt_specific_locals)):
                torch.save(FeatExt_specific_locals[i].state_dict(), os.path.join(model_save_path,
                    "FeatExt_specific"+str(i)+"-{}.pt".format(epoch+1)))
                torch.save(Decoder_locals[i].state_dict(), os.path.join(model_save_path,
                    "Decoder"+str(i)+"-{}.pt".format(epoch+1)))




