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
acc = 0.8
n_cache = 30
n_obs = 10000

n_iter = 2
alpha = 0.9

KL_ind = np.zeros((n_sim,n_rounds+1))
KL_coop = np.zeros((n_sim,n_rounds+1))
KL_feed = np.zeros((n_sim,n_rounds+1))
KL_unif = np.zeros((n_sim,n_rounds+1))
KL_lwb = np.zeros((n_sim,n_rounds+1))

x_hat_dist = np.ones((n_device,n_class))/n_class
sim_bar = tqdm(total=n_sim)
sim_i = 0
while sim_i<n_sim:
    if 1:
        sim_i +=1
        np.random.seed(sim_i)
        d_size_0 = 10000

        D_0_ind,D_0_coop,D_0_feed = generate_D0(n_class,d_size_0)
        D_0_lwb = np.array(D_0_ind,int)
        D_0_unif = np.array(D_0_ind,int)

        
        D_target = np.ones((n_class,1))*d_size_0/n_class

        obs_class = []

        for i in range(n_device):
            #obs_class.append(random.sample(range(n_class),random.randint(int(n_class/2),n_class)))
            obs_class.append(random.sample(range(n_class),n_class))


        x_dist, N_x = create_xdist(n_device,n_class,obs_class,n_obs)

        #x_dist,N_x = generate_xdist_Nx(n_device,n_class,n_obs)

        P_cond,P_condr,P_Condr = generate_cond_probs(n_class,n_device,x_dist,acc)

        Xs,X_samples,N_x_hat = generate_Ys_Y_hat(P_cond,n_device,n_class,n_obs,N_x)

        #x_hat_dist = estimate_xpred(N_x_hat,P_cond,n_device,n_class,x_hat_dist,alpha)

        #print(x_hat_dist,x_dist)

        KL_ind[sim_i-1,0] = cp.norm(D_0_ind-D_target).value/sum(D_target)
        KL_coop[sim_i-1,0] = cp.norm(D_0_coop-D_target).value/sum(D_target)
        KL_feed[sim_i-1,0] = cp.norm(D_0_feed-D_target).value/sum(D_target)
        KL_unif[sim_i-1,0] = cp.norm(D_0_unif-D_target).value/sum(D_target)
        KL_lwb[sim_i-1,0] = cp.norm(D_0_lwb-D_target).value/sum(D_target)

        for round_i in range(n_rounds):
            
            D_target = np.ones((n_class,1))*(d_size_0 + n_cache*n_device)/n_class

            D_0_ind, A_ind = Int_Ind_Update_norm(n_device,n_class,D_target,P_condr,D_0_ind,n_cache,d_size_0,Xs,X_samples,N_x_hat)
            D_0_coop, A_coop = Int_Coop_Update_norm(n_device,n_class,D_target,P_Condr,D_0_coop,n_cache,d_size_0,Xs,X_samples,N_x_hat)
            _, A_ind_feed = Int_Ind_Update_norm(n_device,n_class,D_target,P_condr,D_0_feed,n_cache,d_size_0,Xs,X_samples,N_x_hat)
            D_0_feed, A_feed = Int_Feed_Update_norm(n_device,n_class,D_target,P_condr,D_0_feed,n_cache,d_size_0,A_ind_feed,n_iter,Xs,X_samples,N_x_hat)
            D_0_unif, A_unif = Int_Unif_Update(n_device,n_class,D_0_unif,n_cache,d_size_0,Xs,X_samples,N_x_hat)
            D_0_lwb,_ = Int_Lwb_Update_norm(n_device,n_class,D_target,P_Condr,D_0_lwb,n_cache,d_size_0)
            

            d_size_0 += n_cache*n_device
            
            KL_ind[sim_i-1,round_i+1] = cp.norm(D_0_ind-D_target).value/sum(D_target)
            KL_coop[sim_i-1,round_i+1] = cp.norm(D_0_coop-D_target).value/sum(D_target)
            KL_feed[sim_i-1,round_i+1] = cp.norm(D_0_feed-D_target).value/sum(D_target)
            KL_unif[sim_i-1, round_i+1] = cp.norm(D_0_unif-D_target).value/sum(D_target)
            KL_lwb[sim_i-1, round_i+1] = cp.norm(D_0_lwb-D_target).value/sum(D_target)
        sim_bar.update(1) 
    else:
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

coop_df.to_csv("data/Int_Coop_L2.csv")
ind_df.to_csv("data/Int_Ind_L2.csv")
feed_df.to_csv("data/Int_Feed_L2.csv")
unif_df.to_csv("data/Int_Unif_L2.csv")
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
plt.savefig("plots/int_action.jpg")
