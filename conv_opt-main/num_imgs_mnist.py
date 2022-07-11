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

n_sim = 2
n_class = 10
#n_interval = 20
#n_total = 30
num_epoch = 200
lr = 0.01

list_nums = [10,30,100,300,1000,3000,10000]

X_train,X_test,X_val,y_train,y_test,y_val = load_MNIST_dataset("./MNIST/data",0.1)

Accs = np.zeros((n_sim,len(list_nums)))

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

test_dataset = create_MNIST_dataset(X_test,y_test)

for j in range(n_sim):
    for k,n_i in enumerate(list_nums):

        # Creating Dataset

        #Xs,ys = X_train[:(n_i+1)*n_interval].clone().detach(),y_train[:(n_i+1)*n_interval].clone().detach()
        Xs,ys = X_train[:n_i].clone().detach(),y_train[:n_i].clone().detach()

        x_dist = np.ones((n_class))

        x_dist = x_dist / sum(x_dist)

        N_x = list(map(int,saferound(x_dist*n_i,places=0)))
        ind = 0
        for i in range(n_class):
            ind_s = random.sample(np.argwhere(y_train == i)[0].tolist(), k = N_x[i])
            Xs[ind:ind+N_x[i],:] = X_train[ind_s,:].clone().detach()
            ys[ind:ind+N_x[i]] = y_train[ind_s].clone().detach()
            ind += N_x[i]
        
        dataset = create_MNIST_dataset(Xs,ys)
        
        # Training Model
        losses = []
        base_model = MNISTClassifier().to(device)
        base_model.apply(init_weights)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
        lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

        dataloader, data, label = create_MNIST_dataloader(dataset , n_i, device)

        losses = train_MNIST_model(base_model,losses,loss_fn,optimizer,num_epoch,data,label,dataloader,True)

        test_matrix = test_MNIST_function(base_model,test_dataset,device)

        test_accs = acc_calc(test_matrix)

        print("Test Accuracy:",'{0:.3g}'.format(test_accs))

        Accs[j,k] = test_accs


acc_df = pd.DataFrame(Accs.reshape(-1,1),columns=["Acc"])
acc_df["Training Dataset Size"] = list_nums*n_sim

acc_df.to_csv("data/num_img_mnist.csv")

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
plt.savefig("plots/num_imgs_mnist.jpg")