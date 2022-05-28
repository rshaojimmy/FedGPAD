import os
import os.path as osp
import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from core import Test, Train_FedAvgDep
from datasets.DatasetLoader import get_dataset_loader
from datasets.TargetDatasetLoader import get_tgtdataset_loader
from misc.utils import init_model, init_random_seed, mkdirs
from misc.saver import Saver
from models import DGFANet 
import random

def main(args):

	datasetlistname = '-'.join(args.datasetlist_train)

	if 'print' in args.attacktypelist or 'video' in args.attacktypelist:
		attacktypelistname = '-'.join(args.attacktypelist)
		savefilename = osp.join(args.traintype, datasetlistname+'-'+attacktypelistname+'-'+args.trainidx)
	else:
		savefilename = osp.join(args.traintype, datasetlistname+args.trainidx)


	if args.run_type is 'Train':
		print('datasetlistname: {}'.format(datasetlistname))
		summary_writer = SummaryWriter(osp.join(args.results_path, 'log', savefilename))
		saver = Saver(args,savefilename)
		saver.print_config()

	##################### load seed#####################  
	args.seed = init_random_seed(args.manual_seed)

	#####################load datasets##################### 

	if args.run_type is 'Train':
		data_loader_real_list = []
		data_loader_fake_list = []
		data_loader_test_list = []

		for i in range(len(args.datasetlist_train)):
			data_loader_real = get_dataset_loader(args=args, name=args.datasetlist_train[i], 
								getreal=True, batch_size=args.batchsize)
			data_loader_fake = get_dataset_loader(args=args, name=args.datasetlist_train[i], 
								getreal=False, batch_size=args.batchsize,attacktype=args.attacktypelist[i])

			data_loader_real_list.append(data_loader_real)
			data_loader_fake_list.append(data_loader_fake)
			

	elif args.run_type in ['Test']:
		data_loader_train_list = []
		data_loader_test_list = []

		for i in range(len(args.datasetlist_train)):
			data_loader_real = get_dataset_loader(args=args, name=args.datasetlist_train[i], 
								getreal=True, batch_size=args.batchsize)
			data_loader_train_list.append(data_loader_real)
			data_loader_fake = get_dataset_loader(args=args, name=args.datasetlist_train[i], 
								getreal=False, batch_size=args.batchsize,attacktype=args.attacktypelist[i])
			data_loader_train_list.append(data_loader_fake)

		for i in range(len(args.datasetlist_test)):
			data_loader_target = get_tgtdataset_loader(args=args, name=args.datasetlist_test[i], batch_size=args.batchsize)
			data_loader_test_list.append(data_loader_target)


	##################### load models##################### 

	if args.run_type is 'Train':
     
		if args.traintype is 'FedAvgDep':
			DepthEst = DGFANet.DepthEstmator()
			FeatExt = DGFANet.FeatExtractor()
			Clsfier = DGFANet.Classifier()
			Decoder = DGFANet.Decoder(concat_operation=args.concat_operation)

			DepthEst = init_model(net=DepthEst, init_type = args.init_type, init=True, restore=None)
			FeatExt = init_model(net=FeatExt, init_type = args.init_type, init=True, restore=None)
			Clsfier = init_model(net=Clsfier, init_type = args.init_type, init=True, restore=None)
			Decoder = init_model(net=Decoder, init_type = args.init_type, init=True, restore=None)

			Train_FedAvgDep(args, FeatExt, Clsfier, DepthEst, Decoder,  
				   data_loader_real_list, data_loader_fake_list,
				   summary_writer, saver, savefilename) 



	elif args.run_type is 'Test':

		if args.traintype is 'FedAvgDep':
			FeatExt = DGFANet.FeatExtractor()
			Clsfier = DGFANet.Classifier()

			FeatExt_restore = osp.join(args.results_path, 'snapshots', savefilename, 'FeatExt-'+args.snapshotnum+'.pt')
			Clsfier_restore = osp.join(args.results_path, 'snapshots', savefilename, 'Clsfier-'+args.snapshotnum+'.pt')

			FeatExt = init_model(net=FeatExt, init_type = args.init_type, init=True, restore=FeatExt_restore)
			Clsfier = init_model(net=Clsfier, init_type = args.init_type, init=True, restore=Clsfier_restore)

			Test(args, FeatExt, Clsfier,
				data_loader_test_list, data_loader_train_list,
				savefilename)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="FL_FAS")
	# 2D: 'OULU' 'MSU' 'idiap' 'CASIA' 'SiW'  3D: 'HKBUMARsV2' '3DMAD'
	parser.add_argument('--datasetlist_train', type=str, default=['OULU','CASIA','MSU']) 
	parser.add_argument('--datasetlist_test', type=str, default=['idiap'])
	# 2D: 'print' 'video' 'all'
	parser.add_argument('--attacktypelist', type=str, default=['all','all','all']) 
	# model
	parser.add_argument('--init_type', type=str, default='xavier')# xavier normal   
	# optimizer
	parser.add_argument('--lr', type=float, default=1e-2)  
	parser.add_argument('--optimizer', type=str, default='adam') 

	# # # # training configs
	parser.add_argument('--concat_operation', type=str, default='add') #cat add catrelu addrelu mul
	parser.add_argument('--net_type', type=str, default='DepthAux')
	parser.add_argument('--run_type', type=str, default='Test') # Train Test
	parser.add_argument('--traintype', type=str, default='FedAvgDep') # SF FedAvgDep 
	parser.add_argument('--results_path', type=str, default='./results/Train_xxxxx')
	parser.add_argument('--batchsize', type=int, default=16) 
	parser.add_argument('--w_dep', type=int, default=10) 
	parser.add_argument('--w_diff', type=int, default=1) 
	parser.add_argument('--w_rec', type=int, default=0.1)  
	parser.add_argument('--eps', type=float, default=1e-6)  

	parser.add_argument('--snapshotnum', type=str, default='10')    
	parser.add_argument('--trainidx', type=str, default='1')       

	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--log_step', type=int, default=10)
	parser.add_argument('--model_save_epoch', type=int, default=1)
	parser.add_argument('--manual_seed', type=int, default=None)
 
	print(parser.parse_args())
	main(parser.parse_args())

