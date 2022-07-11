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
from synt_model import Classifier, init_weights

n_sim = 10
n_rounds = 5
n_class = 15
n_device =20
n_obs = 1000
n_cache = 2
n_iter = 1
num_epoch = 400
b_size = 200
n_size = 60
train_base = False
create_data = True
train_rounds = True
train_base_sim = True
n_features = 2

D_target = np.ones((n_class,1))/n_class

KL_ind = np.zeros((n_sim,n_rounds+1))
KL_coop = np.zeros((n_sim,n_rounds+1))
KL_feed = np.zeros((n_sim,n_rounds+1))
KL_unif = np.zeros((n_sim,n_rounds+1))
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
y_pred_ind = dict()
y_pred_feed = dict()
y_pred_coop = dict()
y_pred_unif = dict()


N_y_pred_ind = np.zeros((n_device,n_class),int)
N_y_pred_feed = np.zeros((n_device,n_class),int)
N_y_pred_coop = np.zeros((n_device,n_class),int)
N_y_pred_unif = np.zeros((n_device,n_class),int)

num_epoch  = 100
base_num_epoch = 100
b_size = 10

ind_acc = np.zeros((n_sim,n_rounds+1))
feed_acc = np.zeros((n_sim,n_rounds+1))
coop_acc = np.zeros((n_sim,n_rounds+1))
unif_acc = np.zeros((n_sim,n_rounds+1))

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

        base_model_ind = Classifier(n_features,n_class).to(device)
        base_model_feed = Classifier(n_features,n_class).to(device)
        base_model_coop = Classifier(n_features,n_class).to(device)
        base_model_unif = Classifier(n_features,n_class).to(device)
            
        base_model_coop.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_ind.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_feed.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        base_model_unif.load_state_dict(torch.load("./synthetic/base_model/basemodel.pt"))
        
        if train_rounds:

            optimizer_coop = optim.SGD(base_model_coop.parameters(), lr=lr,weight_decay=0.01)
            optimizer_feed = optim.SGD(base_model_feed.parameters(), lr=lr,weight_decay=0.01)
            optimizer_ind = optim.SGD(base_model_ind.parameters(), lr=lr,weight_decay=0.01)
            optimizer_unif = optim.SGD(base_model_unif.parameters(), lr=lr,weight_decay=0.01)

            lr_sch_ind = lr_scheduler.ExponentialLR(optimizer_ind,gamma=0.9,last_epoch=-1)
            lr_sch_coop = lr_scheduler.ExponentialLR(optimizer_coop,gamma=0.9,last_epoch=-1)
            lr_sch_feed = lr_scheduler.ExponentialLR(optimizer_feed,gamma=0.9,last_epoch=-1)
            lr_sch_unif = lr_scheduler.ExponentialLR(optimizer_unif,gamma=0.9,last_epoch=-1)

        P_cond_ind = cond_preb_calc(base_model_ind,val_dataset,device ,n_class,n_features)
        P_cond_coop = cond_preb_calc(base_model_coop,val_dataset,device ,n_class,n_features)
        P_cond_feed = cond_preb_calc(base_model_feed,val_dataset,device ,n_class,n_features)
        P_cond_unif = cond_preb_calc(base_model_unif,val_dataset,device ,n_class,n_features)

        dataset_ind = dataset_0.clone().detach()
        dataset_coop = dataset_0.clone().detach()
        dataset_feed = dataset_0.clone().detach()
        dataset_unif = dataset_0.clone().detach()

        D_0 = dataset_stat(dataset_0,n_class,n_features)

        obs_class = []

        for i in range(n_device):
            #obs_class.append(random.sample(range(n_class),random.randint(1,n_class)))
            obs_class.append(random.sample(range(n_class),n_class))

        x_dist, N_x = create_xdist(n_device,n_class,obs_class,n_obs)
        
        D_0_ind = np.array(D_0,int)
        D_0_coop = np.array(D_0,int)
        D_0_feed = np.array(D_0,int)
        D_0_unif = np.array(D_0,int)
        D_0_lwb = np.array(D_0,int)

        if train_rounds:
            test_matrix_ind = test_function(base_model_ind,test_dataset,n_class,n_features,device)
            test_accs_ind = acc_calc(test_matrix_ind)
                
            test_matrix_coop = test_function(base_model_coop,test_dataset,n_class,n_features,device)
            test_accs_coop = acc_calc(test_matrix_coop)

            test_matrix_feed = test_function(base_model_feed,test_dataset,n_class,n_features,device)
            test_accs_feed = acc_calc(test_matrix_feed)

            test_matrix_unif = test_function(base_model_unif,test_dataset,n_class,n_features,device)
            test_accs_unif = acc_calc(test_matrix_unif)

            ind_acc[sim_i-1,0] = test_accs_ind
            feed_acc[sim_i-1,0] = test_accs_feed
            coop_acc[sim_i-1,0] = test_accs_coop
            unif_acc[sim_i-1,0] = test_accs_unif
            
        
        KL_ind[sim_i-1,0] = cp.norm(D_0_ind-D_target).value/sum(D_target)
        KL_coop[sim_i-1,0] = cp.norm(D_0_coop-D_target).value/sum(D_target)
        KL_feed[sim_i-1,0] = cp.norm(D_0_feed-D_target).value/sum(D_target)
        KL_unif[sim_i-1,0] = cp.norm(D_0_unif-D_target).value/sum(D_target)
        KL_lwb[sim_i-1,0] = cp.norm(D_0_lwb-D_target).value/sum(D_target)

        for round_i in range(n_rounds):

            D_target = np.ones((n_class,1))*(sum(D_target) + n_cache*n_device)/n_class
            
            for i in range(n_device):

                obs_datasets[i] = create_obs_datasets(N_x[i],n_obs,X_train,y_train,n_features)

                y_pred_ind[i] = eval_obs(base_model_ind,obs_datasets[i],device,n_features)
                y_pred_coop[i] = eval_obs(base_model_coop,obs_datasets[i],device,n_features)
                y_pred_feed[i] = eval_obs(base_model_feed,obs_datasets[i],device,n_features)
                y_pred_unif[i] = eval_obs(base_model_unif,obs_datasets[i],device,n_features)

                N_y_pred_ind[i,:] = y_pred_stats(y_pred_ind[i],n_class).reshape(-1)
                N_y_pred_coop[i,:] = y_pred_stats(y_pred_coop[i],n_class).reshape(-1)
                N_y_pred_feed[i,:] = y_pred_stats(y_pred_feed[i],n_class).reshape(-1)
                N_y_pred_unif[i,:] = y_pred_stats(y_pred_unif[i],n_class).reshape(-1)

            A_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_ind,D_0_ind,n_cache,x_dist,N_y_pred_ind)
            A_coop = Int_Coop_Action_norm(n_device,n_class,D_target,P_cond_coop,D_0_coop,n_cache,x_dist,N_y_pred_coop)
            A_ind_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,x_dist,N_y_pred_feed)
            A_feed = Int_Feed_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,A_ind_feed,n_iter,x_dist,N_y_pred_feed)
            A_unif = Int_Unif_Action(n_device,n_class,n_cache,N_y_pred_unif)
            D_0_lwb = Int_Lwb_Action_norm(n_device,n_class,D_target,D_0_lwb,n_cache)

            cached_ind = cache_imgs(A_ind,obs_datasets,y_pred_ind,n_cache,n_device,n_class,n_features)
            cached_coop = cache_imgs(A_coop,obs_datasets,y_pred_coop,n_cache,n_device,n_class,n_features)
            cached_feed = cache_imgs(A_feed,obs_datasets,y_pred_feed,n_cache,n_device,n_class,n_features)
            cached_unif = cache_imgs(A_unif,obs_datasets,y_pred_unif,n_cache,n_device,n_class,n_features)

            dataset_coop = torch.cat((dataset_coop,cached_coop))
            dataset_ind = torch.cat((dataset_ind,cached_ind))
            dataset_feed = torch.cat((dataset_feed,cached_feed))
            dataset_unif = torch.cat((dataset_unif,cached_unif))

            if train_rounds:
                losses = []

                dataloader_ind, data_ind, label_ind = create_dataloader(dataset_ind , b_size, device, n_features)
                _ = train_model(base_model_ind,losses,loss_fn,optimizer_ind,num_epoch,data_ind,label_ind,n_features,dataloader_ind,True)

                dataloader_feed, data_feed, label_feed = create_dataloader(dataset_feed , b_size, device, n_features)
                _ = train_model(base_model_feed,losses,loss_fn,optimizer_feed,num_epoch,data_feed,label_feed,n_features,dataloader_feed,True)

                dataloader_coop, data_coop, label_coop = create_dataloader(dataset_coop , b_size, device, n_features)
                _ = train_model(base_model_coop,losses,loss_fn,optimizer_coop,num_epoch,data_coop,label_coop,n_features,dataloader_coop,True)

                dataloader_unif, data_unif, label_unif = create_dataloader(dataset_unif , b_size, device, n_features)
                _ = train_model(base_model_unif,losses,loss_fn,optimizer_unif,num_epoch,data_unif,label_unif,n_features,dataloader_unif,True)

                test_matrix_ind = test_function(base_model_ind,test_dataset,n_class,n_features,device)
                test_accs_ind = acc_calc(test_matrix_ind)

                test_matrix_coop = test_function(base_model_coop,test_dataset,n_class,n_features,device)
                test_accs_coop = acc_calc(test_matrix_coop)

                test_matrix_feed = test_function(base_model_feed,test_dataset,n_class,n_features,device)
                test_accs_feed = acc_calc(test_matrix_feed)

                test_matrix_unif = test_function(base_model_unif,test_dataset,n_class,n_features,device)
                test_accs_unif = acc_calc(test_matrix_unif)

                ind_acc[sim_i-1,round_i+1] = test_accs_ind
                feed_acc[sim_i-1,round_i+1] = test_accs_feed
                coop_acc[sim_i-1,round_i+1] = test_accs_coop
                unif_acc[sim_i-1,round_i+1] = test_accs_unif

                print(test_accs_feed - test_accs_ind)

            else:
                ind_acc[sim_i-1,round_i+1] = ind_acc[sim_i-1,round_i]
                feed_acc[sim_i-1,round_i+1] = feed_acc[sim_i-1,round_i]
                coop_acc[sim_i-1,round_i+1] = coop_acc[sim_i-1,round_i]
                unif_acc[sim_i-1,round_i+1] = unif_acc[sim_i-1,round_i]

            D_0_ind = dataset_stat(dataset_ind,n_class,n_features)
            D_0_coop = dataset_stat(dataset_coop,n_class,n_features)
            D_0_feed = dataset_stat(dataset_feed,n_class,n_features)
            D_0_unif = dataset_stat(dataset_unif,n_class,n_features)

            KL_ind[sim_i-1,round_i+1] = cp.norm(D_0_ind-D_target).value/sum(D_target)
            KL_coop[sim_i-1,round_i+1] = cp.norm(D_0_coop-D_target).value/sum(D_target)
            KL_feed[sim_i-1,round_i+1] = cp.norm(D_0_feed-D_target).value/sum(D_target)
            KL_unif[sim_i-1, round_i+1] = cp.norm(D_0_unif-D_target).value/sum(D_target)
            KL_lwb[sim_i-1,round_i+1] = cp.norm(D_0_lwb-D_target).value/sum(D_target)

        sim_bar.update(1) 
    else:
        sim_i -= 1
        print(sim_i)

sim_bar.close()

if train_rounds:
    ind_acc_df = pd.DataFrame(ind_acc.reshape(-1,1),columns=["Acc"])
    ind_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    coop_acc_df = pd.DataFrame(coop_acc.reshape(-1,1),columns=["Acc"])
    coop_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed_acc_df = pd.DataFrame(feed_acc.reshape(-1,1),columns=["Acc"])
    feed_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    unif_acc_df = pd.DataFrame(unif_acc.reshape(-1,1),columns=["Acc"])
    unif_acc_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    
    coop_acc_df.to_csv("data/Synt_Coop_Acc.csv")
    ind_acc_df.to_csv("data/Synt_Ind_Acc.csv")
    feed_acc_df.to_csv("data/Synt_Feed_Acc.csv")
    unif_acc_df.to_csv("data/Synt_Unif_Acc.csv")

    fig, ax = plt.subplots(figsize=(7,7),dpi=600)

    sns.lineplot(data=coop_acc_df,x="Round",y="Acc",label="Cooperative",linewidth=3,linestyle='-')
    sns.lineplot(data=ind_acc_df,x="Round",y="Acc",label="Individual",linewidth=3,linestyle='-.')
    sns.lineplot(data=feed_acc_df,x="Round",y="Acc",label="Interactive",linewidth=3,linestyle='--')
    sns.lineplot(data=unif_acc_df,x="Round",y="Acc",label="Uniform",linewidth=3,linestyle='-')

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/sim_synt_acc.jpg")

    print(np.mean(coop_acc,0)[-1])
    print(np.mean(feed_acc,0)[-1])
    print(np.mean(ind_acc,0)[-1])
    print(np.mean(unif_acc,0)[-1])
else:

    coop_df = pd.DataFrame(KL_coop.reshape(-1,1),columns=["L2"])
    coop_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    ind_df = pd.DataFrame(KL_ind.reshape(-1,1),columns=["L2"])
    ind_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    feed_df = pd.DataFrame(KL_feed.reshape(-1,1),columns=["L2"])
    feed_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    unif_df = pd.DataFrame(KL_unif.reshape(-1,1),columns=["L2"])
    unif_df["Round"] = [i for i in range(n_rounds+1)]*n_sim
    lwb_df = pd.DataFrame(KL_lwb.reshape(-1,1),columns=["L2"])
    lwb_df["Round"] = [i for i in range(n_rounds+1)]*n_sim

    coop_df.to_csv("data/Synt_Coop_L2.csv")
    ind_df.to_csv("data/Synt_Ind_L2.csv")
    feed_df.to_csv("data/Synt_Feed_L2.csv")
    unif_df.to_csv("data/Synt_Unif_L2.csv")
    lwb_df.to_csv("data/Int_Lwb_L2.csv")

    fig, ax = plt.subplots(figsize=(7,7),dpi=600)

    sns.lineplot(data=coop_df,x="Round",y="L2",label="Oracle",linewidth=3,linestyle='-')
    sns.lineplot(data=ind_df,x="Round",y="L2",label="Greedy",linewidth=3,linestyle='-.')
    sns.lineplot(data=unif_df,x="Round",y="L2",label="Uniform",linewidth=3,linestyle='-')
    sns.lineplot(data=feed_df,x="Round",y="L2",label="Interactive",linewidth=3,linestyle='--')
    sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')

    plt.grid(linestyle='--', linewidth=2)
    plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
    plt.ylabel("L2 Norm",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/sim_synt_action.jpg")
