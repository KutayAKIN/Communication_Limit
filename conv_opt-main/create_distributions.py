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

dataset_type = "AdverseWeather"
sim_loc = "./combined/AdverseWeather/runs/run26"

if dataset_type == "MNIST":
    X_train,X_test,X_val,y_train,y_test,y_val = load_MNIST_dataset("./MNIST/data",0.1)
elif dataset_type == "CIFAR10":
    X_train,X_test,X_val,y_train,y_test,y_val = load_CIFAR10_dataset("./CIFAR10/data",0.1)
elif dataset_type == "CIFAR100":
    X_train,X_test,X_val,y_train,y_test,y_val = load_CIFAR100_dataset("./CIFAR100/data",0.1)
elif dataset_type == "AdverseWeather":
    X_train,X_test,X_val,y_train,y_test,y_val,n_class = load_AdverseWeather_labels("./AdverseWeather/weather_labels.yml","./AdverseWeather/daytime_labels.yml",0.1,0.1)
elif dataset_type == "BDD":
    X_train,X_test,X_val,y_train,y_test,y_val = load_BDD_labels("/home/oa5983/conv_opt/conv_opt/BDD/bdd100k/labels/det_20",
    "/home/oa5983/conv_opt/conv_opt/BDD/bdd100k/images/100k",0.1)
else:
    assert 1==1, "wrong dataset type"

n_class = len(set(y_train.tolist()))

with open(sim_loc+"/feed_dataset_ind.yml","r") as f:
    feed_dataset_ind = yaml.safe_load(f)

with open(sim_loc+"/ind_dataset_ind.yml","r") as f:
    ind_dataset_ind = yaml.safe_load(f)

sim_is = list(feed_dataset_ind.keys())

init_sims = np.zeros((len(sim_is),n_class))

final_feed_sims = np.zeros((len(sim_is),n_class))

final_ind_sims = np.zeros((len(sim_is),n_class))

for i,sim_i in enumerate(sim_is):

    init_sims[i] = np.sort(y_pred_stats(y_train[feed_dataset_ind[sim_i][0]].numpy(),n_class).reshape(-1))[::-1]

    final_feed_sims[i] = np.sort(y_pred_stats(y_train[feed_dataset_ind[sim_i][-1]].numpy(),n_class).reshape(-1))[::-1]

    final_ind_sims[i] = np.sort(y_pred_stats(y_train[ind_dataset_ind[sim_i][-1]].numpy(),n_class).reshape(-1))[::-1]

init_mean = np.sum(init_sims,0)/ np.sum(init_sims)

final_feed_mean = np.sum(final_feed_sims,0)/ np.sum(final_feed_sims)

final_ind_mean = np.sum(final_ind_sims,0)/ np.sum(final_ind_sims)

stats_df = pd.DataFrame(np.concatenate((init_mean,final_ind_mean,final_feed_mean)),columns=["Dist"])

stats_df["Class"] = [i for i in range(n_class)]*3

stats_df["Policy"] = ["Initial"]*n_class + ["Greedy"]*n_class + ["Interactive"]*n_class 

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(7,7),dpi=600)

gfg = sns.barplot(data=stats_df,x="Class",y="Dist",hue="Policy",palette=["b","r","g"])

# for legend text
plt.setp(gfg.get_legend().get_texts(), fontsize='14') 
 
# for legend title
plt.setp(gfg.get_legend().get_title(), fontsize='18') 

plt.ylabel("Class Occurance",fontweight="bold" ,fontsize=24)
plt.xlabel("Classes",fontweight="bold" ,fontsize=24)
plt.rcParams["font.size"]=18
plt.rcParams["axes.linewidth"]=2
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.tight_layout()
plt.savefig(sim_loc+"/dists.jpg")


