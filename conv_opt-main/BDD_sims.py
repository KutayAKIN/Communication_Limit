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

    X_train,X_test,X_val,y_train,y_test,y_val,n_class = load_BDD_labels(opt.label_loc,opt.image_loc,0.1,opt.cache_all)

    val_dataset = create_AdverseWeather_dataset(X_val,y_val,opt.cache_all)

    test_dataset = create_AdverseWeather_dataset(X_test,y_test,opt.cache_all)

    params = dict()

    params["n_device"] = opt.n_device
    params["n_sim"] = 1
    params["n_rounds"] = opt.n_rounds
    params["n_epoch"] = opt.n_epoch
    params["b_size"] = opt.b_size
    params["n_iter"] = opt.n_iter
    params["n_class"] = n_class
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

            Unif_Model = AdverseWeather_Unif_Sim(params,device)
            Ind_Model = AdverseWeather_Ind_Sim(params,device)
            Feed_Model = AdverseWeather_Feed_Sim(params,device)
            Coop_Model = AdverseWeather_Coop_Sim(params,device)
            Lwb_Model = AdverseWeather_Lwb_Sim(params,device)

            Unif_Model.create_base_inds(y_train,base_classes,sim_i,sim_i)
            if opt.cache_all:
                initial_dataset = Unif_Model.create_traindataset(X_train[Unif_Model.dataset_ind[sim_i]],y_train[Unif_Model.dataset_ind[sim_i]],opt.cache_all)
            else:
                initial_dataset = Unif_Model.create_traindataset([X_train[i] for i in Unif_Model.dataset_ind[sim_i][0]],y_train[Unif_Model.dataset_ind[sim_i]],opt.cache_all)

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

            Unif_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist,opt.cache_all)
            Ind_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist,opt.cache_all)
            Feed_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist,opt.cache_all)
            Coop_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist,opt.cache_all)
            Lwb_Model.sim_round(0,sim_i,X_train,y_train,val_dataset,Unif_Model.dataset_ind[sim_i],obs_inds,x_dist)

            Unif_Model.sim_acc(X_train,y_train,test_dataset,opt.cache_all)
            Ind_Model.sim_acc(X_train,y_train,test_dataset,opt.cache_all)
            Feed_Model.sim_acc(X_train,y_train,test_dataset,opt.cache_all)
            Coop_Model.sim_acc(X_train,y_train,test_dataset,opt.cache_all)

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

        except:
            print("Error in sim "+ str(sim_i))
if __name__ == "__main__":

    # Dataset Parameters for the simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-loc", type=str,default="/home/oa5983/conv_opt/conv_opt/BDD/bdd100k/labels/det_20")
    parser.add_argument("--image-loc", type=str,default="/home/oa5983/conv_opt/conv_opt/BDD/bdd100k/images/100k")
    parser.add_argument("--device-no", type=int,default=2)
    parser.add_argument("--n-device", type=int, default=10)
    parser.add_argument("--n-sim", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--n-epoch", type=int, default=50)
    parser.add_argument("--init-sim", type=int, default=0) 
    parser.add_argument("--b-size", type=int, default=128)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--test-b-size", type=int, default=128)
    parser.add_argument("--lr", type=int, default=0.1)
    parser.add_argument("--n-size", type=int, default=2000)
    parser.add_argument("--n-obs", type=int, default=2000)
    parser.add_argument("--n-cache", type=int, default=20)
    parser.add_argument("--cache-all", type=bool, default=True)
    parser.add_argument("--run-loc", type=str, default="./runs/BDD")

    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.device_no) if (torch.cuda.is_available()) else "cpu")
    
    opt.run_loc = opt.run_loc+"/device"+str(opt.device_no)
    
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.device_no) if torch.cuda.is_available() else 'CPU'))

    run_sim(opt,device)
