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

from utils import *
from synt_model import  init_weights
from MNIST_model import MNISTClassifier

X_train,X_test,X_val,y_train,y_test,y_val = load_CIFAR100_dataset("./CIFAR100/data",0.1)

val_dataset = create_CIFAR10_dataset(X_val,y_val)

test_dataset = create_CIFAR10_dataset(X_test,y_test)

params = dict()

params["n_device"] = 20
params["n_sim"] = 1
params["n_rounds"] = 5
params["n_epoch"] = 100
params["b_size"] = 1024
params["n_iter"] = 1
params["n_class"] = 100
params["test_b_size"] = 1000
params["lr"] = 0.05
params["n_size"] = 30000
params["n_obs"] = 2000
params["n_cache"] = 80

device = torch.device("cuda:6" if (torch.cuda.is_available()) else "cpu")

run_loc = "./runs/CIFAR100"

sim_i = 0
random.seed(sim_i)
torch.manual_seed(sim_i)
np.random.seed(sim_i)

base_classes =  [i for i in range(params["n_class"])]

Unif_Model = CIFAR100_Unif_Sim(params,device)
Unif_Model.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True,verbose=False).to(device)  
    
Unif_Model.create_unif_base_inds(y_train,base_classes,sim_i,sim_i)

initial_dataset = Unif_Model.create_traindataset(X_train[Unif_Model.dataset_ind[sim_i]],y_train[Unif_Model.dataset_ind[sim_i]])

Unif_Model.train_model(initial_dataset,False)

test_matrix,labels_stats,top5accs = Unif_Model.test_model(test_dataset,True)
accs = Unif_Model.acc_calc(test_matrix,labels_stats)

print("Test Accuracy:",'{0:.3g}'.format(accs))
print("Test Accuracy top5:",'{0:.3g}'.format(top5accs))