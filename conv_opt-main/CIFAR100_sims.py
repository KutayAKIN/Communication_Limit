import numpy as np
import cvxpy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import yaml
import os, sys
import torch

from utils import *
from synt_model import  init_weights
from MNIST_model import MNISTClassifier

def run_sim(opt,device):

    X_train,X_test,X_val,y_train,y_test,y_val = load_CIFAR100_dataset(opt.dataset_loc,0.1)

    val_dataset = create_CIFAR10_dataset(X_val,y_val)

    test_dataset = create_CIFAR10_dataset(X_test,y_test)

    params = dict()

    params["n_device"] = opt.n_device
    params["n_sim"] = 1
    params["n_rounds"] = opt.n_rounds
    params["n_epoch"] = opt.n_epoch
    params["b_size"] = opt.b_size
    params["n_iter"] = opt.n_iter
    params["n_class"] = opt.n_class
    params["test_b_size"] = opt.test_b_size
    params["lr"] = opt.lr
    params["n_size"] = opt.n_size
    params["n_obs"] = opt.n_obs
    params["n_cache"] = opt.n_cache

    run_loc = opt.run_loc
    
    sim_bar = tqdm(range(opt.init_sim, opt.init_sim+opt.n_sim),total=opt.n_sim)

    for sim_i in sim_bar:

        try:

            random.seed(sim_i)
            torch.manual_seed(sim_i)
            np.random.seed(sim_i)

            run_i_loc = create_run_dir(run_loc)

            base_classes =  [i for i in range(params["n_class"])]

            Unif_Model = CIFAR100_Unif_Sim(params,device)
            Ind_Model = CIFAR100_Ind_Sim(params,device)
            Feed_Model = CIFAR100_Feed_Sim(params,device)
            Coop_Model = CIFAR100_Coop_Sim(params,device)
            Lwb_Model = CIFAR100_Lwb_Sim(params,device)

            Unif_Model.create_base_inds(y_train,base_classes,sim_i,sim_i)

            initial_dataset = Unif_Model.create_traindataset(X_train[Unif_Model.dataset_ind[sim_i]],y_train[Unif_Model.dataset_ind[sim_i]])

            Unif_Model.train_model(initial_dataset)

            test_matrix,labels_stats = Unif_Model.test_model(test_dataset)
            accs = Unif_Model.acc_calc(test_matrix,labels_stats)

            print("Test Accuracy:",'{0:.3g}'.format(accs))

            Unif_Model.save_model(run_i_loc,"init_model"+str(sim_i)+".pt")

            obs_class = [base_classes for i in range(params["n_device"])]

            Unif_Model.load_model(run_i_loc+"/init_model"+str(sim_i)+".pt")
            Ind_Model.load_model(run_i_loc+"/init_model"+str(sim_i)+".pt")
            Feed_Model.load_model(run_i_loc+"/init_model"+str(sim_i)+".pt")
            Coop_Model.load_model(run_i_loc+"/init_model"+str(sim_i)+".pt")

            x_dist, N_x = Unif_Model.create_xdist(sim_i,obs_class,y_train)
            
            obs_inds = Unif_Model.create_obs_ind(N_x,y_train,sim_i)

            Unif_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist)
            Ind_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist)
            Feed_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist)
            Coop_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist)
            Lwb_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist)

            Unif_Model.sim_acc(X_train,y_train,test_dataset)
            Ind_Model.sim_acc(X_train,y_train,test_dataset)
            Feed_Model.sim_acc(X_train,y_train,test_dataset)
            Coop_Model.sim_acc(X_train,y_train,test_dataset)

            Unif_Model.save_model(run_i_loc,"unif_last_model"+str(sim_i)+".pt")
            Ind_Model.save_model(run_i_loc,"ind_last_model"+str(sim_i)+".pt")
            Feed_Model.save_model(run_i_loc,"feed_last_model"+str(sim_i)+".pt")
            Coop_Model.save_model(run_i_loc,"coop_last_model"+str(sim_i)+".pt")

            Unif_Model.save_infos(run_i_loc,"unif")
            Ind_Model.save_infos(run_i_loc,"ind")
            Feed_Model.save_infos(run_i_loc,"feed")
            Coop_Model.save_infos(run_i_loc,"coop")
            Lwb_Model.save_infos(run_i_loc,"lwb")

            plot_KLs(Coop_Model.KLs,Ind_Model.KLs,Feed_Model.KLs,Unif_Model.KLs,Lwb_Model.KLs,run_i_loc+"/L2_Norm.jpg")

            plot_Accs(Coop_Model.accs,Ind_Model.accs,Feed_Model.accs,Unif_Model.accs,run_i_loc+"/Accs.jpg")
            plot_Accs(Coop_Model.min_accs,Ind_Model.min_accs,Feed_Model.min_accs,Unif_Model.min_accs,run_i_loc+"/min_Accs.jpg")
            plot_Accs(Coop_Model.top5accs,Ind_Model.top5accs,Feed_Model.top5accs,Unif_Model.top5accs,run_i_loc+"/top5_Accs.jpg")
        except:
            print("Error in sim "+ str(sim_i))
if __name__ == "__main__":

    # Dataset Parameters for the simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-loc", type=str,default="./CIFAR100/data")
    parser.add_argument("--device-no", type=int,default=0)
    parser.add_argument("--n-device", type=int, default=20)
    parser.add_argument("--n-sim", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--n-epoch", type=int, default=200)
    parser.add_argument("--b-size", type=int, default=1000)
    parser.add_argument("--init-sim", type=int, default=0) 
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--n-class", type=int, default=100)
    parser.add_argument("--test-b-size", type=int, default=1000)
    parser.add_argument("--lr", type=int, default=0.1)
    parser.add_argument("--n-size", type=int, default=20000)
    parser.add_argument("--n-obs", type=int, default=2000)
    parser.add_argument("--n-cache", type=int, default=100)
    parser.add_argument("--run-loc", type=str, default="./runs/CIFAR100")

    opt = parser.parse_args()


    device = torch.device("cuda:"+str(opt.device_no) if (torch.cuda.is_available()) else "cpu")
    
    opt.run_loc = opt.run_loc+"/device"+str(opt.device_no)
    
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.device_no) if torch.cuda.is_available() else 'CPU'))

    run_sim(opt,device)




