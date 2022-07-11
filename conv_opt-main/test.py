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

from utilsmy import *
from synt_model import Classifier, init_weights

n_sim = 10
n_rounds = 1
n_class = 15
n_device =200
n_obs = 1000
n_cache = 2
n_iter = 1
num_epoch = 400
b_size = 200
n_size = 60
train_base = False
create_data = True
train_rounds = False#True
train_base_sim = True
n_features = 2

D_target = np.ones((n_class,1))/n_class

KL_feed10 = np.zeros((n_sim,n_rounds+1))
KL_feed20 = np.zeros((n_sim,n_rounds+1))
KL_feed = np.zeros((n_sim,n_rounds+1))
KL_feed50 = np.zeros((n_sim,n_rounds+1))
KL_lwb = np.zeros((n_sim,n_rounds+1))

if create_data:
    create_synt_dataset("./synthetic/data",n_features,n_class,2000,0.8,0.2)

X_train,X_test,X_val,y_train,y_test,y_val = load_synt_dataset("./synthetic/data")

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

base_classes = [i for i in range(n_class)]

base_model = Classifier(n_features,n_class).to(device)
base_model.apply(init_weights)

filt = [True if i in base_classes else False for i in y_train]

train_dataset = create_dataset(X_train[filt],y_train[filt],n_features)

val_dataset = create_dataset(X_val,y_val,n_features)

test_dataset = create_dataset(X_test,y_test,n_features)

loss_fn = nn.CrossEntropyLoss()
lr = 0.1
optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

if train_base and not train_base_sim:

    losses = []

    dataloader, data, label = create_dataloader(train_dataset , b_size, device, n_features)

    losses = train_model(base_model,losses,loss_fn,optimizer,num_epoch,data,label,n_features,dataloader,False)

    test_matrix = test_function(base_model,test_dataset,n_class,n_features,device)

    test_accs = acc_calc(test_matrix)

    print("Test Accuracy:",'{0:.3g}'.format(test_accs))

    isExist = os.path.exists("./synthetic/base_model")

    if not isExist:
        os.makedirs("./synthetic/base_model")

    torch.save(base_model.state_dict(), "./synthetic/base_model/basemodel.pt")

obs_datasets = dict()
y_pred_feed10 = dict()
y_pred_feed = dict()
y_pred_feed20 = dict()
y_pred_feed50 = dict()


N_y_pred_feed10 = np.zeros((n_device,n_class),int)
N_y_pred_feed = np.zeros((n_device,n_class),int)
N_y_pred_feed20 = np.zeros((n_device,n_class),int)
N_y_pred_feed50 = np.zeros((n_device,n_class),int)

num_epoch  = 100
base_num_epoch = 100
b_size = 10

feed10_acc = np.zeros((n_sim,n_rounds+1))
feed_acc = np.zeros((n_sim,n_rounds+1))
feed20_acc = np.zeros((n_sim,n_rounds+1))
feed50_acc = np.zeros((n_sim,n_rounds+1))

sim_bar = tqdm(total=n_sim)
sim_i = 0



random.seed(sim_i)
np.random.seed(sim_i)
if train_base_sim:
    
    base_classes = random.sample(range(n_class),n_class)

    dataset_0 = create_base_dataset(X_train,y_train,base_classes,n_size,n_features)

    losses = []

    base_model = Classifier(n_features,n_class).to(device)
    base_model.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
    lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

    dataloader, data, label = create_dataloader(dataset_0 , b_size, device, n_features)

    losses = train_model(base_model,losses,loss_fn,optimizer,base_num_epoch,data,label,n_features,dataloader,True)

    test_matrix = test_function(base_model,test_dataset,n_class,n_features,device)

    test_accs = acc_calc(test_matrix)

    print("Test Accuracy:",'{0:.3g}'.format(test_accs))

    isExist = os.path.exists("./synthetic/base_model")

    if not isExist:
        os.makedirs("./synthetic/base_model")

    torch.save(base_model.state_dict(), "./synthetic/base_model/basemodel.pt")
else:

    dataset_0 = create_base_dataset(X_train,y_train,base_classes,n_size,n_features)

D_target = np.ones((n_class,1))*n_size/n_class

sim_i +=1

base_model_feed10 = Classifier(n_features,n_class).to(device)
base_model_feed = Classifier(n_features,n_class).to(device)
base_model_feed20 = Classifier(n_features,n_class).to(device)
base_model_feed50 = Classifier(n_features,n_class).to(device)
    
base_model_feed20.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
base_model_feed10.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
base_model_feed.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
base_model_feed50.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))

if train_rounds:

    optimizer_feed20 = optim.SGD(base_model_feed20.parameters(), lr=lr,weight_decay=0.01)
    optimizer_feed = optim.SGD(base_model_feed.parameters(), lr=lr,weight_decay=0.01)
    optimizer_feed10 = optim.SGD(base_model_feed10.parameters(), lr=lr,weight_decay=0.01)
    optimizer_feed50 = optim.SGD(base_model_feed50.parameters(), lr=lr,weight_decay=0.01)

    lr_sch_feed10 = lr_scheduler.ExponentialLR(optimizer_feed10,gamma=0.9,last_epoch=-1)
    lr_sch_feed20 = lr_scheduler.ExponentialLR(optimizer_feed20,gamma=0.9,last_epoch=-1)
    lr_sch_feed = lr_scheduler.ExponentialLR(optimizer_feed,gamma=0.9,last_epoch=-1)
    lr_sch_feed50 = lr_scheduler.ExponentialLR(optimizer_feed50,gamma=0.9,last_epoch=-1)

P_cond_feed10 = cond_preb_calc(base_model_feed10,val_dataset,device ,n_class,n_features)
P_cond_feed20 = cond_preb_calc(base_model_feed20,val_dataset,device ,n_class,n_features)
P_cond_feed = cond_preb_calc(base_model_feed,val_dataset,device ,n_class,n_features)
P_cond_feed50 = cond_preb_calc(base_model_feed50,val_dataset,device ,n_class,n_features)

dataset_feed10 = dataset_0.clone().detach()
dataset_feed20 = dataset_0.clone().detach()
dataset_feed = dataset_0.clone().detach()
dataset_feed50 = dataset_0.clone().detach()

D_0 = dataset_stat(dataset_0,n_class,n_features)

obs_class = []

for i in range(n_device):
    #obs_class.append(random.sample(range(n_class),random.randint(1,n_class)))
    obs_class.append(random.sample(range(n_class),n_class))

x_dist, N_x = create_xdist(n_device,n_class,obs_class,n_obs)

D_0_feed10 = np.array(D_0,int)
D_0_feed20 = np.array(D_0,int)
D_0_feed = np.array(D_0,int)
D_0_feed50 = np.array(D_0,int)
D_0_lwb = np.array(D_0,int)

if train_rounds:
    test_matrix_feed10 = test_function(base_model_feed10,test_dataset,n_class,n_features,device)
    test_accs_feed10 = acc_calc(test_matrix_feed10)
        
    test_matrix_feed20 = test_function(base_model_feed20,test_dataset,n_class,n_features,device)
    test_accs_feed20 = acc_calc(test_matrix_feed20)

    test_matrix_feed = test_function(base_model_feed,test_dataset,n_class,n_features,device)
    test_accs_feed = acc_calc(test_matrix_feed)

    test_matrix_feed50 = test_function(base_model_feed50,test_dataset,n_class,n_features,device)
    test_accs_feed50 = acc_calc(test_matrix_feed50)

    feed10_acc[sim_i-1,0] = test_accs_feed10
    feed_acc[sim_i-1,0] = test_accs_feed
    feed20_acc[sim_i-1,0] = test_accs_feed20
    feed50_acc[sim_i-1,0] = test_accs_feed50
    

KL_feed10[sim_i-1,0] = cp.norm(D_0_feed10-D_target).value/sum(D_target)
KL_feed20[sim_i-1,0] = cp.norm(D_0_feed20-D_target).value/sum(D_target)
KL_feed[sim_i-1,0] = cp.norm(D_0_feed-D_target).value/sum(D_target)
KL_feed50[sim_i-1,0] = cp.norm(D_0_feed50-D_target).value/sum(D_target)
KL_lwb[sim_i-1,0] = cp.norm(D_0_lwb-D_target).value/sum(D_target)

for round_i in range(n_rounds):

    D_target = np.ones((n_class,1))*(sum(D_target) + n_cache*n_device)/n_class
    
    for i in range(n_device):

        obs_datasets[i] = create_obs_datasets(N_x[i],n_obs,X_train,y_train,n_features)

        y_pred_feed10[i] = eval_obs(base_model_feed10,obs_datasets[i],device,n_features)
        y_pred_feed20[i] = eval_obs(base_model_feed20,obs_datasets[i],device,n_features)
        y_pred_feed[i] = eval_obs(base_model_feed,obs_datasets[i],device,n_features)
        y_pred_feed50[i] = eval_obs(base_model_feed50,obs_datasets[i],device,n_features)
        
        N_y_pred_feed10[i,:] = y_pred_stats(y_pred_feed10[i],n_class).reshape(-1)
        N_y_pred_feed20[i,:] = y_pred_stats(y_pred_feed20[i],n_class).reshape(-1)
        N_y_pred_feed[i,:] = y_pred_stats(y_pred_feed[i],n_class).reshape(-1)
        N_y_pred_feed50[i,:] = y_pred_stats(y_pred_feed50[i],n_class).reshape(-1)

    A_feed10_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed10,D_0_feed10,n_cache,x_dist,N_y_pred_feed10)
    A_feed20_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed20,D_0_feed20,n_cache,x_dist,N_y_pred_feed20)
    A_feed50_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed50,D_0_feed50,n_cache,x_dist,N_y_pred_feed50)
    A_feed10 = Int_Feed_Action_norm_my10(n_device,n_class,D_target,P_cond_feed10,D_0_feed10,n_cache,A_feed10_feed,n_iter,x_dist,N_y_pred_feed10)
    A_feed20 = Int_Feed_Action_norm_my20(n_device,n_class,D_target,P_cond_feed20,D_0_feed20,n_cache,A_feed20_feed,n_iter,x_dist,N_y_pred_feed20)
    A_feed_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,x_dist,N_y_pred_feed)
    A_feed = Int_Feed_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,A_feed_feed,n_iter,x_dist,N_y_pred_feed)
    A_feed50 = Int_Feed_Action_norm_my50(n_device,n_class,D_target,P_cond_feed50,D_0_feed50,n_cache,A_feed50_feed,n_iter,x_dist,N_y_pred_feed50)
    D_0_lwb = Int_Lwb_Action_norm(n_device,n_class,D_target,D_0_lwb,n_cache)

    cached_feed10 = cache_imgs(A_feed10,obs_datasets,y_pred_feed10,n_cache,n_device,n_class,n_features)
    cached_feed20 = cache_imgs(A_feed20,obs_datasets,y_pred_feed20,n_cache,n_device,n_class,n_features)
    cached_feed = cache_imgs(A_feed,obs_datasets,y_pred_feed,n_cache,n_device,n_class,n_features)
    cached_feed50 = cache_imgs(A_feed50,obs_datasets,y_pred_feed50,n_cache,n_device,n_class,n_features)

    dataset_feed20 = torch.cat((dataset_feed20,cached_feed20))
    dataset_feed10 = torch.cat((dataset_feed10,cached_feed10))
    dataset_feed = torch.cat((dataset_feed,cached_feed))
    dataset_feed50 = torch.cat((dataset_feed50,cached_feed50))

    print(P_cond_feed)
    print(np.sum(P_cond_feed, axis=0))
    print(np.sum(P_cond_feed, axis=1))




