import numpy as np
import cvxpy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
from iteround import saferound
import random
import seaborn as sns
import pandas as pd

from utils import *


n_sim = 10
n_rounds = 10
n_class = 20
n_device = 30
n_cache = 20
D_target = np.ones((n_class,1))/n_class
n_iter = 1

KL_ind = np.zeros((n_sim,n_rounds+1))
KL_coop = np.zeros((n_sim,n_rounds+1))
KL_feed = np.zeros((n_sim,n_rounds+1))
KL_unif = np.zeros((n_sim,n_rounds+1))
KL_lwb = np.zeros((n_sim,n_rounds+1))

sim_bar = tqdm(total=n_sim)
sim_i = 0
P_cond = np.zeros((n_device,n_class,n_class))

while sim_i<n_sim:
    try:
        np.random.seed(sim_i)
        sim_i +=1

        d_size_0 = 10000
        D_0 = np.random.rand(n_class,1)
        D_0 = D_0 * d_size_0 / np.sum(D_0)

        D_0_ind = np.array(D_0)
        D_0_coop = np.array(D_0)
        D_0_feed = np.array(D_0)
        D_0_unif = np.array(D_0)
        D_0_lwb = np.array(D_0)
        D_target = np.ones((n_class,1))*d_size_0/n_class
        

        x_dist = np.random.rand(n_device,n_class)
        x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    #x_dist = np.random.rand(1,n_class) 
    #x_dist = np.ones(1,n_class)
    #x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)
    #x_dist = x_dist.repeat(n_device,axis=0)

    #P_cond = np.random.rand(n_device,n_class,n_class)    
    #P_cond = P_cond /np.sum(P_cond,2).reshape(n_device,-1,1)
    
    #P_cond = np.random.rand(1,n_class,n_class)
        P_cond = np.eye(n_class).reshape(1,n_class,n_class)
        P_cond = P_cond /np.sum(P_cond,2).reshape(1,-1,1)
        P_cond = P_cond.repeat(n_device,axis=0)


        P_occ = x_dist.reshape(n_device,1,-1) * P_cond
        P_condr = P_occ /np.sum(P_occ,1).reshape(n_device,1,-1)

        P_tuple = [P_condr[i] for i in range(n_device)]
        P_Condr = np.concatenate(P_tuple,axis=1)

        KL_ind[sim_i-1,0] = cp.norm(D_0_ind-D_target).value/sum(D_target)
        KL_coop[sim_i-1,0] = cp.norm(D_0_coop-D_target).value/sum(D_target)
        KL_feed[sim_i-1,0] = cp.norm(D_0_feed-D_target).value/sum(D_target)
        KL_unif[sim_i-1,0] = cp.norm(D_0_unif-D_target).value/sum(D_target)
        KL_lwb[sim_i-1,0] = cp.norm(D_0_lwb-D_target).value/sum(D_target)
        

        for round_i in range(n_rounds):

            D_target = np.ones((n_class,1))*(d_size_0 + n_cache*n_device)/n_class

            D_0_coop, _ = Cont_Coop_Update_norm(n_device,n_class,D_target,P_Condr,D_0_coop,n_cache,d_size_0)
            D_0_ind, _ = Cont_Ind_Update_norm(n_device,n_class,D_target,P_condr,D_0_ind,n_cache,d_size_0)
            _, A_ind_feed = Cont_Ind_Update_norm(n_device,n_class,D_target,P_condr,D_0_feed,n_cache,d_size_0)
            D_0_feed, _ = Cont_Feed_Update_norm(n_device,n_class,D_target,P_condr,D_0_feed,n_cache,d_size_0,A_ind_feed,n_iter)
            D_0_unif,_ = Cont_Unif_Update_norm(n_device,n_class,P_condr,D_0_unif,n_cache,d_size_0)
            D_0_lwb,_ = Cont_Lwb_Update_norm(n_device,n_class,D_target,P_Condr,D_0_lwb,n_cache,d_size_0)
            

            KL_ind[sim_i-1,round_i+1] = cp.norm(D_0_ind-D_target).value/sum(D_target)
            KL_coop[sim_i-1,round_i+1] = cp.norm(D_0_coop-D_target).value/sum(D_target)
            KL_feed[sim_i-1,round_i+1] = cp.norm(D_0_feed-D_target).value/sum(D_target)
            KL_unif[sim_i-1, round_i+1] = cp.norm(D_0_unif-D_target).value/sum(D_target)
            KL_lwb[sim_i-1, round_i+1] = cp.norm(D_0_lwb-D_target).value/sum(D_target)

            d_size_0 += n_cache*n_device
        sim_bar.update(1) 
    except:
        sim_i -= 1
        print(sim_i)

sim_bar.close()

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

coop_df.to_csv("data/Cont_Coop_L2.csv")
ind_df.to_csv("data/Cont_Ind_L2.csv")
feed_df.to_csv("data/Cont_Feed_L2.csv")
unif_df.to_csv("data/Cont_Unif_L2.csv")
lwb_df.to_csv("data/Cont_Lwb_L2.csv")


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
plt.savefig("plots/cont_action.jpg")