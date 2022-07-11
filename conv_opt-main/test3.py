import numpy as np
import cvxpy as cp
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from utilsmy import *
from synt_model import  init_weights
from CIFAR10_model import ResNet18_10
import torchvision.models as vsmodels


n_sim = 10
num_epoch = 200
b_size = 50
n_obs = 1000
n_iter = 1
n_rounds = 10
n_class = 10
test_b_size = 100
feder_rounds = 5
params = dict()

n_devices = [4]
n_dev = 4
n_caches = [50]

n_sizes = [200]

X_train,X_test,X_val,y_train,y_test,y_val = load_CIFAR10_dataset("./CIFAR/data",0.1)

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

#dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
#dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
#dict_users_train = iid(dataset_train, n_dev)
#dict_users_test = iid(dataset_test, n_dev)

train_ind,test_ind,val_ind = ind_datasets(y_train,y_test,y_val)

train_dataset = create_CIFAR10_traindataset(X_train,y_train)

val_dataset = create_CIFAR10_dataset(X_val,y_val)

test_dataset = create_CIFAR10_dataset(X_test,y_test)

args = args_parser()
args.dataset = 'cifar10'
args.model = 'cnn'
args.iid = True
args.num_users = n_dev
args.frac = 1
args.local_ep = 15
args.shard_per_user = 2
args.lr = 0.01

m = max(int(args.frac * args.num_users), 1)
idxs_users = np.random.choice(range(args.num_users), m, replace=False)


for n_cache in n_caches:
    for n_size in n_sizes:
        for n_device in n_devices:
            
            KL_ind = np.zeros((n_sim,n_rounds+1))
            KL_coop = np.zeros((n_sim,n_rounds+1))
            KL_feed = np.zeros((n_sim,n_rounds+1))
            KL_unif = np.zeros((n_sim,n_rounds+1))
            KL_lwb = np.zeros((n_sim,n_rounds+1))
            KL_feder = np.zeros((n_sim,n_rounds+1))

            device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
            args.device = device

            base_model = ResNet18_10().to(device)
            base_model.apply(init_weights)
            args = args_parser()
            loss_fn = nn.CrossEntropyLoss()
            lr = 0.01
            optimizer = optim.Adam(base_model.parameters(), lr=lr)
            lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

            base_classes =  random.sample(range(10),10)

            dataset_0,_ = create_CIFAR_base_dataset(X_train,y_train,base_classes,n_size)

            losses = []

            dataloader, data, label = create_CIFAR_dataloader(dataset_0 , b_size, device)

            losses = train_CIFAR_model(base_model,losses,loss_fn,optimizer,num_epoch,data,label,dataloader,True)

            test_matrix = test_CIFAR_function(base_model,test_dataset,device,test_b_size)

            test_accs = acc_calc(test_matrix)

            print("Test Accuracy:",'{0:.3g}'.format(test_accs))

            isExist = os.path.exists("./CIFAR/base_model")

            if not isExist:
                os.makedirs("./CIFAR/base_model")

            torch.save(base_model.state_dict(), "./CIFAR/base_model/basemodel.pt")
            
            P_cond_ind = cond_preb_calc_CIFAR(base_model,val_dataset,device)
            P_cond_coop = P_cond_ind
            P_cond_feed = P_cond_ind
            P_cond_unif = P_cond_ind
            P_cond_feder = P_cond_feed

            obs_datasets = dict()
            y_pred_ind = dict()
            y_pred_feed = dict()
            y_pred_coop = dict()
            y_pred_unif = dict()
            y_pred_feder = dict()

            N_y_pred_ind = np.zeros((n_device,n_class),int)
            N_y_pred_feed = np.zeros((n_device,n_class),int)
            N_y_pred_coop = np.zeros((n_device,n_class),int)
            N_y_pred_unif = np.zeros((n_device,n_class),int)
            N_y_pred_feder = np.zeros((n_device,n_class),int)

            ind_acc = np.zeros((n_sim,1))
            feed_acc = np.zeros((n_sim,1))
            coop_acc = np.zeros((n_sim,1))
            unif_acc = np.zeros((n_sim,1))
            feder_acc = np.zeros((n_sim,1))

            sim_bar = tqdm([i for i in range(n_sim)],total=n_sim)

            for sim_i in sim_bar:
                random.seed(sim_i)
                np.random.seed(sim_i)

                D_target = np.ones((n_class,1))*n_size/n_class

                base_model_ind = ResNet18_10().to(device)
                base_model_feed = ResNet18_10().to(device)
                base_model_coop = ResNet18_10().to(device)
                base_model_unif = ResNet18_10().to(device)
                base_model_feder = ResNet18_10().to(device)

                base_model_coop.load_state_dict(torch.load("./CIFAR/base_model/basemodel.pt"))
                base_model_ind.load_state_dict(torch.load("./CIFAR/base_model/basemodel.pt"))
                base_model_feed.load_state_dict(torch.load("./CIFAR/base_model/basemodel.pt"))
                base_model_unif.load_state_dict(torch.load("./CIFAR/base_model/basemodel.pt"))
                base_model_feder.load_state_dict(torch.load("./CIFAR/base_model/basemodel.pt"))


                optimizer_coop = optim.SGD(base_model_coop.parameters(), lr=lr,weight_decay=0.01)
                optimizer_feed = optim.SGD(base_model_feed.parameters(), lr=lr,weight_decay=0.01)
                optimizer_ind = optim.SGD(base_model_ind.parameters(), lr=lr,weight_decay=0.01)
                optimizer_unif = optim.SGD(base_model_unif.parameters(), lr=lr,weight_decay=0.01)

                lr_sch_ind = lr_scheduler.ExponentialLR(optimizer_ind,gamma=0.9,last_epoch=-1)
                lr_sch_coop = lr_scheduler.ExponentialLR(optimizer_coop,gamma=0.9,last_epoch=-1)
                lr_sch_feed = lr_scheduler.ExponentialLR(optimizer_feed,gamma=0.9,last_epoch=-1)
                lr_sch_unif = lr_scheduler.ExponentialLR(optimizer_unif,gamma=0.9,last_epoch=-1)
                
                dataset_ind = dataset_0
                dataset_coop = dataset_0
                dataset_feed = dataset_0
                dataset_unif = dataset_0
                dataset_feder = dataset_0

                D_0 = dataset_MNIST_stat(dataset_ind,n_class)

                obs_class = []

                for i in range(n_device):
                    obs_class.append(random.sample(range(n_class),10))

                x_dist, N_x = create_xdist(n_device,n_class,obs_class,n_obs)
                
                D_0_ind = np.array(D_0,int)
                D_0_coop = np.array(D_0,int)
                D_0_feed = np.array(D_0,int)
                D_0_unif = np.array(D_0,int)
                D_0_lwb = np.array(D_0,int)
                D_0_feder = np.array(D_0,int)

                KL_ind[sim_i-1,0] = cp.norm(D_0_ind-D_target).value/sum(D_target)
                KL_coop[sim_i-1,0] = cp.norm(D_0_coop-D_target).value/sum(D_target)
                KL_feed[sim_i-1,0] = cp.norm(D_0_feed-D_target).value/sum(D_target)
                KL_unif[sim_i-1,0] = cp.norm(D_0_unif-D_target).value/sum(D_target)
                KL_lwb[sim_i-1,0] = cp.norm(D_0_lwb-D_target).value/sum(D_target)
                KL_feder[sim_i-1,0] = cp.norm(D_0_feder-D_target).value/sum(D_target)
                loss_train = []
                lr = args.lr
                results = []

                for round_i in range(n_rounds):
                    
                    D_target = np.ones((n_class,1))*(sum(D_target) + n_cache*n_device)/n_class
                    
                    for i in range(n_device):

                        obs_datasets[i] = create_CIFAR_obs_datasets(N_x[i],n_obs,X_train,y_train)
                        
                        y_pred_ind[i] = eval_CIFAR_obs(base_model_ind,obs_datasets[i],device)
                        y_pred_coop[i] = eval_CIFAR_obs(base_model_coop,obs_datasets[i],device)
                        y_pred_feed[i] = eval_CIFAR_obs(base_model_feed,obs_datasets[i],device)
                        y_pred_unif[i] = eval_CIFAR_obs(base_model_unif,obs_datasets[i],device)

                        N_y_pred_ind[i,:] = y_pred_stats(y_pred_ind[i],n_class).reshape(-1)
                        N_y_pred_coop[i,:] = y_pred_stats(y_pred_coop[i],n_class).reshape(-1)
                        N_y_pred_feed[i,:] = y_pred_stats(y_pred_feed[i],n_class).reshape(-1)
                        N_y_pred_unif[i,:] = y_pred_stats(y_pred_unif[i],n_class).reshape(-1)
                    
                    for round_feder in range(feder_rounds):
                        w_glob = None
                        loss_locals = []
                        for j in range(n_dev):
                            local = LocalUpdate(args=args, dataset=obs_datasets[j])
                            net_local = copy.deepcopy(base_model_feder)

                            w_local, loss = local.train(net=net_local.to(device), device=device)
                            loss_locals.append(copy.deepcopy(loss))

                            if w_glob is None:
                                w_glob = copy.deepcopy(w_local)
                            else:
                                for k in w_glob.keys():
                                    w_glob[k] += w_local[k]
                        else:
                            continue

                        # update global weights
                        for k in w_glob.keys():
                            w_glob[k] = torch.div(w_glob[k], m)

                        # copy weight to net_glob
                        base_model_feder.load_state_dict(w_glob)

                    for k in range(n_dev):
                        y_pred_feder[k] = eval_CIFAR_obs(base_model_unif,obs_datasets[k],device)
                        N_y_pred_feder[k,:] = y_pred_stats(y_pred_feder[k],n_class).reshape(-1)

                    A_feder = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feder,D_0_feder,n_cache,x_dist,N_y_pred_feder)
                    A_ind = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_ind,D_0_ind,n_cache,x_dist,N_y_pred_ind)
                    A_coop = Int_Coop_Action_norm(n_device,n_class,D_target,P_cond_coop,D_0_coop,n_cache,x_dist,N_y_pred_coop)
                    A_ind_feed = Int_Ind_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,x_dist,N_y_pred_feed)
                    A_feed = Int_Feed_Action_norm(n_device,n_class,D_target,P_cond_feed,D_0_feed,n_cache,A_ind_feed,n_iter,x_dist,N_y_pred_feed)
                    A_unif = Int_Unif_Action(n_device,n_class,n_cache,N_y_pred_unif)
                    D_0_lwb = Int_Lwb_Action_norm(n_device,n_class,D_target,D_0_lwb,n_cache)
                
                    cached_feder = cache_imgs_CIFAR(A_feder,obs_datasets,y_pred_feder,n_cache,n_device,n_class)
                    cached_ind = cache_imgs_CIFAR(A_ind,obs_datasets,y_pred_ind,n_cache,n_device,n_class)
                    cached_coop = cache_imgs_CIFAR(A_coop,obs_datasets,y_pred_coop,n_cache,n_device,n_class)
                    cached_feed = cache_imgs_CIFAR(A_feed,obs_datasets,y_pred_feed,n_cache,n_device,n_class)
                    cached_unif = cache_imgs_CIFAR(A_unif,obs_datasets,y_pred_unif,n_cache,n_device,n_class)

                    dataset_coop = torch.utils.data.ConcatDataset((dataset_coop,cached_coop))
                    dataset_ind = torch.utils.data.ConcatDataset((dataset_ind,cached_ind))
                    dataset_feed = torch.utils.data.ConcatDataset((dataset_feed,cached_feed))
                    dataset_unif = torch.utils.data.ConcatDataset((dataset_unif,cached_unif))
                    dataset_feder = torch.utils.data.ConcatDataset((dataset_feder,cached_feder))

                    if round_i == (n_rounds-1):
                        losses = []

                        dataloader_ind, data_ind, label_ind = create_CIFAR_dataloader(dataset_ind , b_size, device)
                        _ = train_CIFAR_model(base_model_ind,losses,loss_fn,optimizer_ind,num_epoch,data_ind,label_ind,dataloader_ind,True)

                        dataloader_feed, data_feed, label_feed = create_CIFAR_dataloader(dataset_feed , b_size, device)
                        _ = train_CIFAR_model(base_model_feed,losses,loss_fn,optimizer_feed,num_epoch,data_feed,label_feed,dataloader_feed,True)

                        dataloader_coop, data_coop, label_coop = create_CIFAR_dataloader(dataset_coop , b_size, device)
                        _ = train_CIFAR_model(base_model_coop,losses,loss_fn,optimizer_coop,num_epoch,data_coop,label_coop,dataloader_coop,True)

                        dataloader_unif, data_unif, label_unif = create_CIFAR_dataloader(dataset_unif , b_size, device)
                        _ = train_CIFAR_model(base_model_unif,losses,loss_fn,optimizer_unif,num_epoch,data_unif,label_unif,dataloader_unif,True)

                        test_matrix_ind = test_CIFAR_function(base_model_ind,test_dataset,device,test_b_size)
                        test_accs_ind = acc_calc(test_matrix_ind)

                        test_matrix_coop = test_CIFAR_function(base_model_coop,test_dataset,device,test_b_size)
                        test_accs_coop = acc_calc(test_matrix_coop)

                        test_matrix_feed = test_CIFAR_function(base_model_feed,test_dataset,device,test_b_size)
                        test_accs_feed = acc_calc(test_matrix_feed)

                        test_matrix_unif = test_CIFAR_function(base_model_unif,test_dataset,device,test_b_size)
                        test_accs_unif = acc_calc(test_matrix_unif)

                        ind_acc[sim_i] = test_accs_ind
                        feed_acc[sim_i] = test_accs_feed
                        coop_acc[sim_i] = test_accs_coop
                        unif_acc[sim_i] = test_accs_unif

                        print(test_accs_feed - test_accs_ind)
            
                    D_0_ind = dataset_MNIST_stat(dataset_ind,n_class)
                    D_0_coop = dataset_MNIST_stat(dataset_coop,n_class)
                    D_0_feed = dataset_MNIST_stat(dataset_feed,n_class)
                    D_0_unif = dataset_MNIST_stat(dataset_unif,n_class)
                    D_0_feder = dataset_MNIST_stat(dataset_feder,n_class)
                    
                    KL_ind[sim_i-1,round_i+1] = cp.norm(D_0_ind-D_target).value/sum(D_target)
                    KL_coop[sim_i-1,round_i+1] = cp.norm(D_0_coop-D_target).value/sum(D_target)
                    KL_feed[sim_i-1,round_i+1] = cp.norm(D_0_feed-D_target).value/sum(D_target)
                    KL_unif[sim_i-1, round_i+1] = cp.norm(D_0_unif-D_target).value/sum(D_target)
                    KL_lwb[sim_i-1,round_i+1] = cp.norm(D_0_lwb-D_target).value/sum(D_target)
                    KL_feder[sim_i-1,round_i+1] = cp.norm(D_0_feder-D_target).value/sum(D_target)

            acc_df = pd.DataFrame(np.concatenate((ind_acc,coop_acc,feed_acc,unif_acc)),columns=["Accuracy"])
            acc_df["Sampling"] = ["Individual"]*n_sim + ["Cooperative"]*n_sim + ["Interactive"]*n_sim + ["Uniform"]*n_sim
            acc_df.to_csv("data/CIFAR_Acc_3.csv")

            fig, ax = plt.subplots(figsize=(7,7),dpi=600)

            sns.barplot(data=acc_df,x="Sampling",y="Accuracy")

            plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)
            plt.rcParams["font.size"]=18
            plt.rcParams["axes.linewidth"]=2
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
            plt.tight_layout()
            plt.savefig("plots/cifar_acc"+str(n_cache)+str(n_size)+str(n_device)+".jpg")

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
            feder_df = pd.DataFrame(KL_feder.reshape(-1,1),columns=["L2"])
            feder_df["Round"] = [i for i in range(n_rounds+1)]*n_sim

            coop_df.to_csv("data/CIFAR_Coop_L2_3.csv")
            ind_df.to_csv("data/CIFAR_Ind_L2_3.csv")
            feed_df.to_csv("data/CIFAR_Feed_L2_3.csv")
            unif_df.to_csv("data/CIFAR_Unif_L2_3.csv")
            lwb_df.to_csv("data/CIFAR_Lwb_L2_3.csv")
            feder_df.to_csv("data/CIFAR_Feder_L2_3.csv")

            fig, ax = plt.subplots(figsize=(7,7),dpi=600)

            sns.lineplot(data=coop_df,x="Round",y="L2",label="Cooperative",linewidth=3,linestyle='-')
            sns.lineplot(data=ind_df,x="Round",y="L2",label="Individual",linewidth=3,linestyle='-.')
            sns.lineplot(data=feed_df,x="Round",y="L2",label="Interactive",linewidth=3,linestyle='--')
            sns.lineplot(data=unif_df,x="Round",y="L2",label="Uniform",linewidth=3,linestyle='-')
            sns.lineplot(data=lwb_df,x="Round",y="L2",label="Lower-Bound",linewidth=3,linestyle='-.')
            sns.lineplot(data=feder_df,x="Round",y="L2",label="FedAvg",linewidth=3,linestyle='-.')

            plt.grid(linestyle='--', linewidth=2)
            plt.xlabel("Round $i$",fontweight="bold" ,fontsize=24)
            plt.ylabel("L2 Norm",fontweight="bold" ,fontsize=24)
            plt.rcParams["font.size"]=18
            plt.rcParams["axes.linewidth"]=2
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig("plots/sim_cifar_action_3.jpg")

           