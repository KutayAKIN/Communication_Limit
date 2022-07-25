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

from utilsmy2 import *
from synt_model import Classifier, init_weights

n_sim = 10
n_rounds = 10
n_class = 15
n_device =20
n_obs = 1000
n_cache = 2
n_iter = 1
num_epoch = 400
b_size = 200
n_size = 60#kilit nokta
train_base = False
create_data = True
train_rounds = False
train_base_sim = True
n_features = 2

a_m_2 = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
[0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]])

a_m_5 = np.array([[0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],[1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0],
[1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0],[0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,1,0],[0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0],[0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,1],[0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0],
[0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0],[0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1],[0,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
[1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1],[0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
[1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0]])

a_m_10 = np.array([[0,1,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,1,1,1],[1,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,0,0],[0,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,1,1],
[0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0],[1,1,1,0,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1],[1,0,0,0,1,0,0,1,1,0,1,1,1,0,0,1,1,1,0,0],[0,1,1,0,1,0,0,0,0,0,0,1,0,1,1,1,0,1,1,1],
[0,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,1],[1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,1],[0,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,0],[1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,0,1,1,0,1],
[0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,0],[0,0,0,1,0,1,0,1,1,1,0,0,0,1,1,1,1,0,1,0],[0,1,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0],[1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,0,1,0],
[1,0,0,0,0,1,1,0,1,1,0,0,1,0,1,0,1,1,0,1],[0,1,0,1,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1],[1,1,1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,0,0,0],[1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1],
[1,0,1,0,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0]])

D_target = np.ones((n_class,1))/n_class

KL_feed10 = np.zeros((n_sim,n_rounds+1))
KL_feed25 = np.zeros((n_sim,n_rounds+1))
KL_feed = np.zeros((n_sim,n_rounds+1))
KL_feed50 = np.zeros((n_sim,n_rounds+1))
KL_coop = np.zeros((n_sim,n_rounds+1))
KL_lwb = np.zeros((n_sim,n_rounds+1))
KL_coop10 = np.zeros((n_sim,n_rounds+1))
KL_coop25 = np.zeros((n_sim,n_rounds+1))
cost_feed10 = np.zeros((n_sim,n_rounds+1))
cost_feed25 = np.zeros((n_sim,n_rounds+1))
cost_feed = np.zeros((n_sim,n_rounds+1))
cost_feed50 = np.zeros((n_sim,n_rounds+1))
cost_coop = np.zeros((n_sim,n_rounds+1))
cost_coop10 = np.zeros((n_sim,n_rounds+1))
cost_coop25 = np.zeros((n_sim,n_rounds+1))

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
y_pred_feed25 = dict()
y_pred_feed50 = dict()
y_pred_coop = dict()
y_pred_coop10 = dict()
y_pred_coop25 = dict()


N_y_pred_feed10 = np.zeros((n_device,n_class),int)
N_y_pred_feed = np.zeros((n_device,n_class),int)
N_y_pred_feed25 = np.zeros((n_device,n_class),int)
N_y_pred_feed50 = np.zeros((n_device,n_class),int)
N_y_pred_coop = np.zeros((n_device,n_class),int)
N_y_pred_coop10 = np.zeros((n_device,n_class),int)
N_y_pred_coop25 = np.zeros((n_device,n_class),int)

num_epoch  = 100
base_num_epoch = 100
b_size = 10

feed10_acc = np.zeros((n_sim,n_rounds+1))
feed_acc = np.zeros((n_sim,n_rounds+1))
feed25_acc = np.zeros((n_sim,n_rounds+1))
feed50_acc = np.zeros((n_sim,n_rounds+1))
coop_acc = np.zeros((n_sim,n_rounds+1))
coop10_acc = np.zeros((n_sim,n_rounds+1))
coop25_acc = np.zeros((n_sim,n_rounds+1))
C = np.zeros((1,n_device))
for i in range(n_device):
    C[0,i] = 10*np.random.rand(1,1)


sim_bar = tqdm(total=n_sim)
sim_i = 0

while sim_i<n_sim:
    if 1:

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

        base_model_coop = Classifier(n_features,n_class).to(device)
        base_model_coop10 = Classifier(n_features,n_class).to(device)
        base_model_coop25 = Classifier(n_features,n_class).to(device)
        base_model_feed10 = Classifier(n_features,n_class).to(device)
        base_model_feed = Classifier(n_features,n_class).to(device)
        base_model_feed25 = Classifier(n_features,n_class).to(device)
        base_model_feed50 = Classifier(n_features,n_class).to(device)
            
        base_model_feed25.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_feed10.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_feed.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_feed50.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_coop.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_coop10.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_coop25.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        if train_rounds:

            optimizer_feed25 = optim.SGD(base_model_feed25.parameters(), lr=lr,weight_decay=0.01)
            optimizer_feed = optim.SGD(base_model_feed.parameters(), lr=lr,weight_decay=0.01)
            optimizer_feed10 = optim.SGD(base_model_feed10.parameters(), lr=lr,weight_decay=0.01)
            optimizer_feed50 = optim.SGD(base_model_feed50.parameters(), lr=lr,weight_decay=0.01)
            optimizer_coop = optim.SGD(base_model_coop.parameters(), lr=lr,weight_decay=0.01)
            optimizer_coop10 = optim.SGD(base_model_coop10.parameters(), lr=lr,weight_decay=0.01)
            optimizer_coop25 = optim.SGD(base_model_coop25.parameters(), lr=lr,weight_decay=0.01)

            lr_sch_feed10 = lr_scheduler.ExponentialLR(optimizer_feed10,gamma=0.9,last_epoch=-1)
            lr_sch_feed25 = lr_scheduler.ExponentialLR(optimizer_feed25,gamma=0.9,last_epoch=-1)
            lr_sch_feed = lr_scheduler.ExponentialLR(optimizer_feed,gamma=0.9,last_epoch=-1)
            lr_sch_feed50 = lr_scheduler.ExponentialLR(optimizer_feed50,gamma=0.9,last_epoch=-1)
            lr_sch_coop = lr_scheduler.ExponentialLR(optimizer_coop,gamma=0.9,last_epoch=-1)
            lr_sch_coop10 = lr_scheduler.ExponentialLR(optimizer_coop10,gamma=0.9,last_epoch=-1)
            lr_sch_coop25 = lr_scheduler.ExponentialLR(optimizer_coop25,gamma=0.9,last_epoch=-1)

        P_cond_feed10 = cond_preb_calc(base_model_feed10,val_dataset,device ,n_class,n_features)
        P_cond_feed25 = cond_preb_calc(base_model_feed25,val_dataset,device ,n_class,n_features)
        P_cond_feed = cond_preb_calc(base_model_feed,val_dataset,device ,n_class,n_features)
        P_cond_feed50 = cond_preb_calc(base_model_feed50,val_dataset,device ,n_class,n_features)
        P_cond_coop = cond_preb_calc(base_model_coop,val_dataset,device ,n_class,n_features)
        P_cond_coop10 = cond_preb_calc(base_model_coop10,val_dataset,device ,n_class,n_features)
        P_cond_coop25 = cond_preb_calc(base_model_coop25,val_dataset,device ,n_class,n_features)

        dataset_feed10 = dataset_0.clone().detach()
        dataset_feed25 = dataset_0.clone().detach()
        dataset_feed = dataset_0.clone().detach()
        dataset_feed50 = dataset_0.clone().detach()
        dataset_coop = dataset_0.clone().detach()
        dataset_coop10 = dataset_0.clone().detach()
        dataset_coop25 = dataset_0.clone().detach()

        D_0 = dataset_stat(dataset_0,n_class,n_features)

        obs_class = []

        for i in range(n_device):
            #obs_class.append(random.sample(range(n_class),random.randint(1,n_class)))
            obs_class.append(random.sample(range(n_class),n_class-1))

        x_dist, N_x = create_xdist2(n_device,n_class,obs_class,n_obs)
        
        D_0_feed10 = np.array(D_0,int)
        D_0_feed25 = np.array(D_0,int)
        D_0_feed = np.array(D_0,int)
        D_0_feed50 = np.array(D_0,int)
        #D_0_lwb = np.array(D_0,int)
        D_0_coop = np.array(D_0,int)
        D_0_coop10 = np.array(D_0,int)
        D_0_coop25 = np.array(D_0,int)

        if train_rounds:
            test_matrix_feed10 = test_function(base_model_feed10,test_dataset,n_class,n_features,device)
            test_accs_feed10 = acc_calc(test_matrix_feed10)
            
            test_matrix_coop = test_function(base_model_coop,test_dataset,n_class,n_features,device)
            test_accs_coop = acc_calc(test_matrix_coop)

            test_matrix_coop10 = test_function(base_model_coop10,test_dataset,n_class,n_features,device)
            test_accs_coop10 = acc_calc(test_matrix_coop10)

            test_matrix_coop25 = test_function(base_model_coop25,test_dataset,n_class,n_features,device)
            test_accs_coop25 = acc_calc(test_matrix_coop25)

            test_matrix_feed25 = test_function(base_model_feed25,test_dataset,n_class,n_features,device)
            test_accs_feed25 = acc_calc(test_matrix_feed25)

            test_matrix_feed = test_function(base_model_feed,test_dataset,n_class,n_features,device)
            test_accs_feed = acc_calc(test_matrix_feed)

            test_matrix_feed50 = test_function(base_model_feed50,test_dataset,n_class,n_features,device)
            test_accs_feed50 = acc_calc(test_matrix_feed50)

            feed10_acc[sim_i-1,0] = test_accs_feed10
            feed_acc[sim_i-1,0] = test_accs_feed
            feed25_acc[sim_i-1,0] = test_accs_feed25
            feed50_acc[sim_i-1,0] = test_accs_feed50
            coop_acc[sim_i-1,0] = test_accs_coop
            coop10_acc[sim_i-1,0] = test_accs_coop10
            coop25_acc[sim_i-1,0] = test_accs_coop25
            
        
        KL_feed10[sim_i-1,0] = cp.norm(D_0_feed10-D_target).value/sum(D_target)
        KL_feed25[sim_i-1,0] = cp.norm(D_0_feed25-D_target).value/sum(D_target)
        KL_feed[sim_i-1,0] = cp.norm(D_0_feed-D_target).value/sum(D_target)
        KL_feed50[sim_i-1,0] = cp.norm(D_0_feed50-D_target).value/sum(D_target)
        #KL_lwb[sim_i-1,0] = cp.norm(D_0_lwb-D_target).value/sum(D_target)
        KL_coop[sim_i-1,0] = cp.norm(D_0_coop-D_target).value/sum(D_target)
        KL_coop10[sim_i-1,0] = cp.norm(D_0_coop10-D_target).value/sum(D_target)
        KL_coop25[sim_i-1,0] = cp.norm(D_0_coop25-D_target).value/sum(D_target)

        for round_i in range(n_rounds):

            D_target = np.ones((n_class,1))*(sum(D_target) + n_cache*n_device)/n_class
            
            for i in range(n_device):

                obs_datasets[i] = create_obs_datasets(N_x[i],n_obs,X_train,y_train,n_features)

                y_pred_feed10[i] = eval_obs(base_model_feed10,obs_datasets[i],device,n_features)
                y_pred_feed25[i] = eval_obs(base_model_feed25,obs_datasets[i],device,n_features)
                y_pred_feed[i] = eval_obs(base_model_feed,obs_datasets[i],device,n_features)
                y_pred_feed50[i] = eval_obs(base_model_feed50,obs_datasets[i],device,n_features)
                y_pred_coop[i] = eval_obs(base_model_coop,obs_datasets[i],device,n_features)
                y_pred_coop10[i] = eval_obs(base_model_coop10,obs_datasets[i],device,n_features)
                y_pred_coop25[i] = eval_obs(base_model_coop25,obs_datasets[i],device,n_features)
                N_y_pred_feed10[i,:] = y_pred_stats(y_pred_feed10[i],n_class).reshape(-1)
                N_y_pred_feed25[i,:] = y_pred_stats(y_pred_feed25[i],n_class).reshape(-1)
                N_y_pred_feed[i,:] = y_pred_stats(y_pred_feed[i],n_class).reshape(-1)
                N_y_pred_feed50[i,:] = y_pred_stats(y_pred_feed50[i],n_class).reshape(-1)
                N_y_pred_coop[i,:] = y_pred_stats(y_pred_coop[i],n_class).reshape(-1)
                N_y_pred_coop10[i,:] = y_pred_stats(y_pred_coop10[i],n_class).reshape(-1)
                N_y_pred_coop25[i,:] = y_pred_stats(y_pred_coop25[i],n_class).reshape(-1)

            A_coop, y_val_coop = Int_Coop_Action_norm_group2(n_device,n_class,D_target,P_cond_coop,D_0_coop,n_cache,x_dist,N_y_pred_coop, C, k=10)
            A_coop_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_coop,D_0_coop,n_cache,x_dist,N_y_pred_coop)
            A_coop_group = np.zeros((n_device*n_class,1),int)
            for f in range(n_device):
                if y_val_coop[f] == 0:
                    A_coop_group[f*n_class:(f+1)*n_class] = np.zeros((n_class,1),int)#A_coop_ind[f*n_class:(f+1)*n_class]
                else:
                    A_coop_group[f*n_class:(f+1)*n_class] = A_coop[f*n_class:(f+1)*n_class]

            A_coop10, y_val_coop10 = Int_Coop_Action_norm_group2(n_device,n_class,D_target,P_cond_coop10,D_0_coop10,n_cache,x_dist,N_y_pred_coop10, C, k=2)
            A_coop10_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_coop10,D_0_coop10,n_cache,x_dist,N_y_pred_coop10)
            A_coop10_group = np.zeros((n_device*n_class,1),int)
            for f in range(n_device):
                if y_val_coop10[f] == 0:
                    A_coop10_group[f*n_class:(f+1)*n_class] = np.zeros((n_class,1),int)#A_coop10_ind[f*n_class:(f+1)*n_class]
                else:
                    A_coop10_group[f*n_class:(f+1)*n_class] = A_coop10[f*n_class:(f+1)*n_class]

            A_coop25, y_val_coop25 = Int_Coop_Action_norm_group2(n_device,n_class,D_target,P_cond_coop25,D_0_coop25,n_cache,x_dist,N_y_pred_coop25, C, k=5)
            A_coop25_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_coop25,D_0_coop25,n_cache,x_dist,N_y_pred_coop25)
            A_coop25_group = np.zeros((n_device*n_class,1),int)
            for f in range(n_device):
                if y_val_coop25[f] == 0:
                    A_coop25_group[f*n_class:(f+1)*n_class] = np.zeros((n_class,1),int)#A_coop25_ind[f*n_class:(f+1)*n_class]
                else:
                    A_coop25_group[f*n_class:(f+1)*n_class] = A_coop25[f*n_class:(f+1)*n_class]

            A_feed10_feed, y_val_feed10 = Int_Coop_Action_norm_group3(n_device,n_class,D_target,P_cond_feed10,D_0_feed10,n_cache,x_dist,N_y_pred_feed10, C, k=2)
            A_feed10_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed10,D_0_feed10,n_cache,x_dist,N_y_pred_feed10)
            A_feed10 = np.zeros((n_device*n_class,1),int)
            for f in range(n_device):
                if y_val_feed10[f] == 0:
                    A_feed10[f*n_class:(f+1)*n_class] = np.zeros((n_class,1),int)#A_feed10_ind[f*n_class:(f+1)*n_class]
                else:
                    A_feed10[f*n_class:(f+1)*n_class] = A_feed10_feed[f*n_class:(f+1)*n_class]

            A_feed25_feed, y_val_feed25 = Int_Coop_Action_norm_group3(n_device,n_class,D_target,P_cond_feed25,D_0_feed25,n_cache,x_dist,N_y_pred_feed25, C, k=5)
            A_feed25_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed25,D_0_feed25,n_cache,x_dist,N_y_pred_feed25)
            A_feed25 = np.zeros((n_device*n_class,1),int)
            for f in range(n_device):
                if y_val_feed25[f] == 0:
                    A_feed25[f*n_class:(f+1)*n_class] = np.zeros((n_class,1),int)#A_feed25_ind[f*n_class:(f+1)*n_class]
                else:
                    A_feed25[f*n_class:(f+1)*n_class] = A_feed25_feed[f*n_class:(f+1)*n_class]

            A_feed50_feed, y_val_feed50 = Int_Coop_Action_norm_group3(n_device,n_class,D_target,P_cond_feed50,D_0_feed50,n_cache,x_dist,N_y_pred_feed50, C, k=10)
            A_feed50_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed50,D_0_feed50,n_cache,x_dist,N_y_pred_feed50)
            A_feed50 = np.zeros((n_device*n_class,1),int)
            for f in range(n_device):
                if y_val_feed50[f] == 0:
                    A_feed50[f*n_class:(f+1)*n_class] = np.zeros((n_class,1),int)#A_feed50_ind[f*n_class:(f+1)*n_class]
                else:
                    A_feed50[f*n_class:(f+1)*n_class] = A_feed50_feed[f*n_class:(f+1)*n_class]
            #A_feed10_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed10,D_0_feed10,n_cache,x_dist,N_y_pred_feed10)
            #A_feed25_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed25,D_0_feed25,n_cache,x_dist,N_y_pred_feed25)
            #A_feed50_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed50,D_0_feed50,n_cache,x_dist,N_y_pred_feed50)
            #A_feed10 = Int_Feed_Action_norm_my10(n_device,n_class,D_target,P_cond_feed10,D_0_feed10,n_cache,A_feed10_feed,n_iter,x_dist,N_y_pred_feed10, a_m_2)
            #A_feed25 = Int_Feed_Action_norm_my10(n_device,n_class,D_target,P_cond_feed25,D_0_feed25,n_cache,A_feed25_feed,n_iter,x_dist,N_y_pred_feed25, a_m_5)
            A_feed_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,x_dist,N_y_pred_feed)
            A_feed = Int_Feed_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,A_feed_feed,n_iter,x_dist,N_y_pred_feed)
            #A_feed50 = Int_Feed_Action_norm_my10(n_device,n_class,D_target,P_cond_feed50,D_0_feed50,n_cache,A_feed50_feed,n_iter,x_dist,N_y_pred_feed50,a_m_10)
            #D_0_lwb = Int_Lwb_Action_norm(n_device,n_class,D_target,D_0_lwb,n_cache)

            cached_feed10 = cache_imgs_feed(A_feed10,obs_datasets,y_pred_feed10,n_cache,n_device,n_class,n_features,k_num=2)
            cached_feed25 = cache_imgs_feed(A_feed25,obs_datasets,y_pred_feed25,n_cache,n_device,n_class,n_features,k_num=5)
            cached_feed = cache_imgs_feed(A_feed,obs_datasets,y_pred_feed,n_cache,n_device,n_class,n_features,k_num=20)
            cached_feed50 = cache_imgs_feed(A_feed50,obs_datasets,y_pred_feed50,n_cache,n_device,n_class,n_features,k_num=10)
            cached_coop = cache_imgs_feed(A_coop_group,obs_datasets,y_pred_coop,n_cache,n_device,n_class,n_features,k_num=10)
            cached_coop10 = cache_imgs_feed(A_coop10_group,obs_datasets,y_pred_coop10,n_cache,n_device,n_class,n_features,k_num=2)
            cached_coop25 = cache_imgs_feed(A_coop25_group,obs_datasets,y_pred_coop25,n_cache,n_device,n_class,n_features,k_num=5)

            dataset_feed25 = torch.cat((dataset_feed25,cached_feed25))
            dataset_feed10 = torch.cat((dataset_feed10,cached_feed10))
            dataset_feed = torch.cat((dataset_feed,cached_feed))
            dataset_feed50 = torch.cat((dataset_feed50,cached_feed50))
            dataset_coop = torch.cat((dataset_coop,cached_coop))
            dataset_coop10 = torch.cat((dataset_coop10,cached_coop10))
            dataset_coop25 = torch.cat((dataset_coop25,cached_coop25))

            if train_rounds:
                losses = []

                dataloader_feed10, data_feed10, label_feed10 = create_dataloader(dataset_feed10 , b_size, device, n_features)
                _ = train_model(base_model_feed10,losses,loss_fn,optimizer_feed10,num_epoch,data_feed10,label_feed10,n_features,dataloader_feed10,True)

                dataloader_coop, data_coop, label_coop = create_dataloader(dataset_coop , b_size, device, n_features)
                _ = train_model(base_model_coop,losses,loss_fn,optimizer_coop,num_epoch,data_coop,label_coop,n_features,dataloader_coop,True)

                dataloader_coop10, data_coop10, label_coop10 = create_dataloader(dataset_coop10 , b_size, device, n_features)
                _ = train_model(base_model_coop10,losses,loss_fn,optimizer_coop10,num_epoch,data_coop10,label_coop10,n_features,dataloader_coop10,True)

                dataloader_coop25, data_coop25, label_coop25 = create_dataloader(dataset_coop25 , b_size, device, n_features)
                _ = train_model(base_model_coop25,losses,loss_fn,optimizer_coop25,num_epoch,data_coop25,label_coop25,n_features,dataloader_coop25,True)

                dataloader_feed, data_feed, label_feed = create_dataloader(dataset_feed , b_size, device, n_features)
                _ = train_model(base_model_feed,losses,loss_fn,optimizer_feed,num_epoch,data_feed,label_feed,n_features,dataloader_feed,True)

                dataloader_feed25, data_feed25, label_feed25 = create_dataloader(dataset_feed25 , b_size, device, n_features)
                _ = train_model(base_model_feed25,losses,loss_fn,optimizer_feed25,num_epoch,data_feed25,label_feed25,n_features,dataloader_feed25,True)

                dataloader_feed50, data_feed50, label_feed50 = create_dataloader(dataset_feed50 , b_size, device, n_features)
                _ = train_model(base_model_feed50,losses,loss_fn,optimizer_feed50,num_epoch,data_feed50,label_feed50,n_features,dataloader_feed50,True)

                test_matrix_feed10 = test_function(base_model_feed10,test_dataset,n_class,n_features,device)
                test_accs_feed10 = acc_calc(test_matrix_feed10)

                test_matrix_coop = test_function(base_model_coop,test_dataset,n_class,n_features,device)
                test_accs_coop = acc_calc(test_matrix_coop)

                test_matrix_coop10 = test_function(base_model_coop10,test_dataset,n_class,n_features,device)
                test_accs_coop10 = acc_calc(test_matrix_coop10)

                test_matrix_coop25 = test_function(base_model_coop25,test_dataset,n_class,n_features,device)
                test_accs_coop25 = acc_calc(test_matrix_coop25)

                test_matrix_feed25 = test_function(base_model_feed25,test_dataset,n_class,n_features,device)
                test_accs_feed25 = acc_calc(test_matrix_feed25)

                test_matrix_feed = test_function(base_model_feed,test_dataset,n_class,n_features,device)
                test_accs_feed = acc_calc(test_matrix_feed)

                test_matrix_feed50 = test_function(base_model_feed50,test_dataset,n_class,n_features,device)
                test_accs_feed50 = acc_calc(test_matrix_feed50)

                feed10_acc[sim_i-1,round_i+1] = test_accs_feed10
                feed_acc[sim_i-1,round_i+1] = test_accs_feed
                feed25_acc[sim_i-1,round_i+1] = test_accs_feed25
                feed50_acc[sim_i-1,round_i+1] = test_accs_feed50
                coop_acc[sim_i-1,round_i+1] = test_accs_coop
                coop10_acc[sim_i-1,round_i+1] = test_accs_coop10
                coop25_acc[sim_i-1,round_i+1] = test_accs_coop25

                print(test_accs_feed - test_accs_feed10)

            else:
                feed10_acc[sim_i-1,round_i+1] = feed10_acc[sim_i-1,round_i]
                feed_acc[sim_i-1,round_i+1] = feed_acc[sim_i-1,round_i]
                feed25_acc[sim_i-1,round_i+1] = feed25_acc[sim_i-1,round_i]
                feed50_acc[sim_i-1,round_i+1] = feed50_acc[sim_i-1,round_i]
                coop_acc[sim_i-1,round_i+1] = coop_acc[sim_i-1,round_i]
                coop10_acc[sim_i-1,round_i+1] = coop10_acc[sim_i-1,round_i]
                coop25_acc[sim_i-1,round_i+1] = coop25_acc[sim_i-1,round_i]

            D_0_feed10 = dataset_stat(dataset_feed10,n_class,n_features)
            D_0_feed25 = dataset_stat(dataset_feed25,n_class,n_features)
            D_0_feed = dataset_stat(dataset_feed,n_class,n_features)
            D_0_feed50 = dataset_stat(dataset_feed50,n_class,n_features)
            D_0_coop = dataset_stat(dataset_coop,n_class,n_features)
            D_0_coop10 = dataset_stat(dataset_coop10,n_class,n_features)
            D_0_coop25 = dataset_stat(dataset_coop25,n_class,n_features)

            KL_feed10[sim_i-1,round_i+1] = (cp.norm(D_0_feed10-D_target).value/sum(D_target))#+(C@y_val_feed10)*10
            KL_feed25[sim_i-1,round_i+1] = (cp.norm(D_0_feed25-D_target).value/sum(D_target))#+(C@y_val_feed25)*4
            KL_feed[sim_i-1,round_i+1] = (cp.norm(D_0_feed-D_target).value/sum(D_target))#+C@np.ones((n_device,1))
            KL_feed50[sim_i-1, round_i+1] = (cp.norm(D_0_feed50-D_target).value/sum(D_target))#+(C@y_val_feed50)*2
            #KL_lwb[sim_i-1,round_i+1] = cp.norm(D_0_lwb-D_target).value/sum(D_target)
            KL_coop[sim_i-1, round_i+1] = cp.norm(D_0_coop-D_target).value/sum(D_target)#+(C@y_val_coop)*2
            KL_coop10[sim_i-1, round_i+1] = cp.norm(D_0_coop10-D_target).value/sum(D_target)#+(C@y_val_coop10)*10
            KL_coop25[sim_i-1, round_i+1] = cp.norm(D_0_coop25-D_target).value/sum(D_target)#+(C@y_val_coop25)*4

            cost_feed10[sim_i-1,round_i+1] = (C@y_val_feed10)*10
            cost_feed25[sim_i-1,round_i+1] = (C@y_val_feed25)*4
            cost_feed[sim_i-1,round_i+1] = C@np.ones((n_device,1))
            cost_feed50[sim_i-1,round_i+1] = (C@y_val_feed50)*2
            cost_coop[sim_i-1,round_i+1] = (C@y_val_coop)*2
            cost_coop10[sim_i-1,round_i+1] = (C@y_val_coop10)*10
            cost_coop25[sim_i-1,round_i+1] = (C@y_val_coop25)*4


        sim_bar.update(1) 
    else:
        sim_i -= 1
        print(sim_i)

sim_bar.close()

if train_rounds:
    feed10_acc_df = pd.DataFrame(feed10_acc.reshape(-1,1),columns=["Acc"])
    feed10_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed25_acc_df = pd.DataFrame(feed25_acc.reshape(-1,1),columns=["Acc"])
    feed25_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed_acc_df = pd.DataFrame(feed_acc.reshape(-1,1),columns=["Acc"])
    feed_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed50_acc_df = pd.DataFrame(feed50_acc.reshape(-1,1),columns=["Acc"])
    feed50_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop_acc_df = pd.DataFrame(coop_acc.reshape(-1,1),columns=["Acc"])
    coop_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop10_acc_df = pd.DataFrame(coop10_acc.reshape(-1,1),columns=["Acc"])
    coop10_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop25_acc_df = pd.DataFrame(coop25_acc.reshape(-1,1),columns=["Acc"])
    coop25_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    
    feed25_acc_df.to_csv("data/Synt_feed25_Acc_group.csv")
    feed10_acc_df.to_csv("data/Synt_feed10_Acc_group.csv")
    feed_acc_df.to_csv("data/Synt_Feed_Acc_group.csv")
    feed50_acc_df.to_csv("data/Synt_feed50_Acc_group.csv")
    coop_acc_df.to_csv("data/Synt_coop_Acc_group.csv")
    coop10_acc_df.to_csv("data/Synt_coop10_Acc_group.csv")
    coop25_acc_df.to_csv("data/Synt_coop25_Acc_group.csv")

    fig, ax = plt.subplots(figsize=(7,7),dpi=600)

    sns.lineplot(data=feed25_acc_df,x="Round",y="Acc",label="feed4",linewidth=3,linestyle='--')
    sns.lineplot(data=feed10_acc_df,x="Round",y="Acc",label="feed40",linewidth=3,linestyle='--')
    sns.lineplot(data=feed_acc_df,x="Round",y="Acc",label="Interactive",linewidth=3,linestyle='-')
    sns.lineplot(data=feed50_acc_df,x="Round",y="Acc",label="feed25",linewidth=3,linestyle='--')
    sns.lineplot(data=coop_acc_df,x="Round",y="Acc",label="k=10",linewidth=3,linestyle='-')
    sns.lineplot(data=coop10_acc_df,x="Round",y="Acc",label="k=2",linewidth=3,linestyle='-')
    sns.lineplot(data=coop25_acc_df,x="Round",y="Acc",label="k=5",linewidth=3,linestyle='-')

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/sim_synt_acc_group.jpg")

    print(np.mean(feed25_acc,0)[-1])
    print(np.mean(feed_acc,0)[-1])
    print(np.mean(feed10_acc,0)[-1])
    print(np.mean(feed50_acc,0)[-1])
    print(np.mean(coop_acc,0)[-1])
else:

    feed25_df = pd.DataFrame(KL_feed25.reshape(-1,1),columns=["L2"])
    feed25_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed10_df = pd.DataFrame(KL_feed10.reshape(-1,1),columns=["L2"])
    feed10_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed_df = pd.DataFrame(KL_feed.reshape(-1,1),columns=["L2"])
    feed_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed50_df = pd.DataFrame(KL_feed50.reshape(-1,1),columns=["L2"])
    feed50_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    #lwb_df = pd.DataFrame(KL_lwb.reshape(-1,1),columns=["L2"])
    #lwb_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop_df = pd.DataFrame(KL_coop.reshape(-1,1),columns=["L2"])
    coop_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop10_df = pd.DataFrame(KL_coop10.reshape(-1,1),columns=["L2"])
    coop10_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop25_df = pd.DataFrame(KL_coop25.reshape(-1,1),columns=["L2"])
    coop25_df["Round"] = [i for i in range(n_rounds+1)]*n_sim

    feed25_df.to_csv("data/Synt_feed25_L2_group.csv")
    feed10_df.to_csv("data/Synt_feed10_L2_group.csv")
    feed_df.to_csv("data/Synt_Feed_L2_group.csv")
    feed50_df.to_csv("data/Synt_feed50_L2_group.csv")
    #lwb_df.to_csv("data/Int_Lwb_L2_group.csv")
    coop_df.to_csv("data/Synt_coop_L2_group.csv")
    coop10_df.to_csv("data/Synt_coop10_L2_group.csv")
    coop25_df.to_csv("data/Synt_coop25_L2_group.csv")

    #fig, ax = plt.subplots(figsize=(30,20),dpi=600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20), dpi = 800)
    #sns.lineplot(data=feed25_df,x="Round",y="L2",linewidth=3,color='red',linestyle='--')
    #sns.lineplot(data=feed10_df,x="Round",y="L2",linewidth=3,color='blue',linestyle='--')
    #sns.lineplot(data=feed50_df,x="Round",y="L2",linewidth=3,color='green',linestyle='--')
    #sns.lineplot(data=coop10_df,x="Round",y="L2",label="k=2",linewidth=3,color='blue',linestyle='-')
    #sns.lineplot(data=coop25_df,x="Round",y="L2",label="k=5",linewidth=3,color='red',linestyle='-')
    #sns.lineplot(data=coop_df,x="Round",y="L2",label="k=10",linewidth=3,color='green',linestyle='-')
    #sns.lineplot(data=feed_df,x="Round",y="L2",label="k=20",linewidth=3,linestyle='-')
    #sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
    ax1.set_title("Random Selection", fontsize = 40)
    ax2.set_title("Boolean Convex", fontsize = 40)
    plt.subplot(1, 2, 1)
    sns.lineplot(data=feed10_df,x="Round",y="L2",linewidth=3,label="k=2",color='blue',linestyle='--')
    sns.lineplot(data=feed25_df,x="Round",y="L2",linewidth=3,label="k=5",color='red',linestyle='--')
    sns.lineplot(data=feed50_df,x="Round",y="L2",linewidth=3,label="k=10",color='green',linestyle='--')
    sns.lineplot(data=feed_df,x="Round",y="L2",label="k=20",color='black',linewidth=3,linestyle='-')
    #sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
    
    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("L2 Norm",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax1.xaxis.set_tick_params(labelsize=20,width=10)
    ax1.yaxis.set_tick_params(labelsize=20,width=10)
    plt.legend(fontsize=40)
    plt.tight_layout()
    #fig, ax = plt.subplots(figsize=(30,20),dpi=600)
    plt.subplot(1, 2, 2)
    sns.lineplot(data=coop10_df,x="Round",y="L2",label="k=2",linewidth=3,color='blue',linestyle='-')
    sns.lineplot(data=coop25_df,x="Round",y="L2",label="k=5",linewidth=3,color='red',linestyle='-')
    sns.lineplot(data=coop_df,x="Round",y="L2",label="k=10",linewidth=3,color='green',linestyle='-')
    sns.lineplot(data=feed_df,x="Round",y="L2",label="k=20",color='black',linewidth=3,linestyle='-')
    #sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
    #plt.title("Boolean Convex")
    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("L2 Norm",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax2.xaxis.set_tick_params(labelsize=20,width=10)
    ax2.yaxis.set_tick_params(labelsize=20,width=10)
    plt.legend(fontsize=40)
    plt.tight_layout()

    plt.savefig("plots/sim_synt_action_group16.jpg")

    cost_feed25_df = pd.DataFrame(cost_feed25.reshape(-1,1),columns=["L2"])
    cost_feed25_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    cost_feed10_df = pd.DataFrame(cost_feed10.reshape(-1,1),columns=["L2"])
    cost_feed10_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    cost_feed_df = pd.DataFrame(cost_feed.reshape(-1,1),columns=["L2"])
    cost_feed_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    cost_feed50_df = pd.DataFrame(cost_feed50.reshape(-1,1),columns=["L2"])
    cost_feed50_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    #lwb_df = pd.DataFrame(KL_lwb.reshape(-1,1),columns=["L2"])
    #lwb_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    cost_coop_df = pd.DataFrame(cost_coop.reshape(-1,1),columns=["L2"])
    cost_coop_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    cost_coop10_df = pd.DataFrame(cost_coop10.reshape(-1,1),columns=["L2"])
    cost_coop10_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    cost_coop25_df = pd.DataFrame(cost_coop25.reshape(-1,1),columns=["L2"])
    cost_coop25_df["Round"] = [i for i in range(n_rounds+1)]*n_sim

    cost_feed25_df.to_csv("data/Synt_cost_feed25_L2_group.csv")
    cost_feed10_df.to_csv("data/Synt_cost_feed10_L2_group.csv")
    cost_feed_df.to_csv("data/Synt_cost_Feed_L2_group.csv")
    cost_feed50_df.to_csv("data/Synt_cost_feed50_L2_group.csv")
    #lwb_df.to_csv("data/Int_Lwb_L2_group.csv")
    cost_coop_df.to_csv("data/Synt_cost_coop_L2_group.csv")
    cost_coop10_df.to_csv("data/Synt_cost_coop10_L2_group.csv")
    cost_coop25_df.to_csv("data/Synt_cost_coop25_L2_group.csv")

    #fig, ax = plt.subplots(figsize=(30,20),dpi=600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20), dpi = 800)
    #sns.lineplot(data=feed25_df,x="Round",y="L2",linewidth=3,color='red',linestyle='--')
    #sns.lineplot(data=feed10_df,x="Round",y="L2",linewidth=3,color='blue',linestyle='--')
    #sns.lineplot(data=feed50_df,x="Round",y="L2",linewidth=3,color='green',linestyle='--')
    #sns.lineplot(data=coop10_df,x="Round",y="L2",label="k=2",linewidth=3,color='blue',linestyle='-')
    #sns.lineplot(data=coop25_df,x="Round",y="L2",label="k=5",linewidth=3,color='red',linestyle='-')
    #sns.lineplot(data=coop_df,x="Round",y="L2",label="k=10",linewidth=3,color='green',linestyle='-')
    #sns.lineplot(data=feed_df,x="Round",y="L2",label="k=20",linewidth=3,linestyle='-')
    #sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
    ax1.set_title("Random Selection", fontsize = 40)
    ax2.set_title("Boolean Convex", fontsize = 40)
    plt.subplot(1, 2, 1)
    sns.lineplot(data=cost_feed10_df,x="Round",y="L2",linewidth=3,label="k=2",color='blue',linestyle='--')
    sns.lineplot(data=cost_feed25_df,x="Round",y="L2",linewidth=3,label="k=5",color='red',linestyle='--')
    sns.lineplot(data=cost_feed50_df,x="Round",y="L2",linewidth=3,label="k=10",color='green',linestyle='--')
    sns.lineplot(data=cost_feed_df,x="Round",y="L2",label="k=20",color='black',linewidth=3,linestyle='-')
    #sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
    
    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("L2 Norm",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax1.xaxis.set_tick_params(labelsize=20,width=10)
    ax1.yaxis.set_tick_params(labelsize=20,width=10)
    plt.legend(fontsize=40)
    plt.tight_layout()
    #fig, ax = plt.subplots(figsize=(30,20),dpi=600)
    plt.subplot(1, 2, 2)
    sns.lineplot(data=cost_coop10_df,x="Round",y="L2",label="k=2",linewidth=3,color='blue',linestyle='-')
    sns.lineplot(data=cost_coop25_df,x="Round",y="L2",label="k=5",linewidth=3,color='red',linestyle='-')
    sns.lineplot(data=cost_coop_df,x="Round",y="L2",label="k=10",linewidth=3,color='green',linestyle='-')
    sns.lineplot(data=cost_feed_df,x="Round",y="L2",label="k=20",color='black',linewidth=3,linestyle='-')
    #sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
    #plt.title("Boolean Convex")
    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("L2 Norm",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax2.xaxis.set_tick_params(labelsize=20,width=10)
    ax2.yaxis.set_tick_params(labelsize=20,width=10)
    plt.legend(fontsize=40)
    plt.tight_layout()

    plt.savefig("plots/sim_synt_action_group17.jpg")