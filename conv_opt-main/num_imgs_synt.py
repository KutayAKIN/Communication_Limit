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

n_sim = 20
n_class = 15
n_interval = 10
n_total = 15
n_obs = 1000
num_epoch = 500
n_features = 2
lr = 0.1
create_synt_dataset("./synthetic/data", n_features,n_class, n_obs, 0.8, 0.2)
X_train,X_test,X_val,y_train,y_test,y_val = load_synt_dataset("./synthetic/data")

Accs = np.zeros((n_sim,n_total))

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

base_model = Classifier(n_features,n_class).to(device)
base_model.apply(init_weights)

test_dataset = create_dataset(X_test,y_test,n_features)

for j in range(n_sim):
    for n_i in range(n_total):

        # Creating Dataset
        dataset = torch.zeros(((n_i+1)*n_interval,n_features+1))

        x_dist = np.ones((n_class))

        x_dist = x_dist / sum(x_dist)

        N_x = list(map(int,saferound(x_dist*(n_i+1)*n_interval,places=0)))
        ind = 0
        for i in range(n_class):
            ind_s = random.sample(np.argwhere(y_train == i).tolist(), k = N_x[i])
            dataset[ind:ind+N_x[i],:n_features] = torch.tensor(X_train[tuple(ind_s),:]).reshape(-1,n_features)
            dataset[ind:ind+N_x[i],n_features] = i
            ind += N_x[i] 
        
        # Training Model
        losses = []
        base_model = Classifier(n_features,n_class).to(device)
        base_model.apply(init_weights)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
        lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

        dataloader, data, label = create_dataloader(dataset , (n_i+1)*n_interval, device, n_features)

        losses = train_model(base_model,losses,loss_fn,optimizer,num_epoch,data,label,n_features,dataloader,True)

        test_matrix = test_function(base_model,test_dataset,n_class,n_features,device)

        test_accs = acc_calc(test_matrix)

        print("Test Accuracy:",'{0:.3g}'.format(test_accs))

        Accs[j,n_i] = test_accs


acc_df = pd.DataFrame(Accs.reshape(-1,1),columns=["Acc"])
acc_df["Training Dataset Size"] = [(i+1)*n_interval for i in range(n_total)]*n_sim

acc_df.to_csv("data/num_img_synt.csv")

fig, ax = plt.subplots(figsize=(7,7),dpi=600)

sns.lineplot(data=acc_df,x="Training Dataset Size",y="Acc",linewidth=3,linestyle='-')
plt.grid(linestyle='--', linewidth=2)
plt.xlabel("Training Dataset Size",fontweight="bold" ,fontsize=24)
plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)
plt.rcParams["font.size"]=18
plt.rcParams["axes.linewidth"]=2
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("plots/num_imgs_synt.jpg")



