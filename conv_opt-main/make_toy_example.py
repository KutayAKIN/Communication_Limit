import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n = 100
N_cache = np.array([[0,1],[1,0]])*n
D_0 = np.array([0,0])*n
D_target = np.array([[0,0],[1.2,1.2]])*n

action_space_1 = np.array([[0.9,0.1],[0.7,0.3]])*n
action_space_2 = np.array([[0.5,0.5],[0.2,0.8]])*n

greedy_action_1 = np.array([[0,0],[0.7,0.3]])*n
greedy_action_2 = np.array([[0,0],[0.5,0.5]])*n
total_greedy_action = greedy_action_1 + greedy_action_2

oracle_action_1 = np.array([[0,0],[0.8,0.2]])*n
oracle_action_2 = np.array([[0,0],[0.2,0.8]])*n
total_oracle_action = oracle_action_1 + oracle_action_2

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(7,7),dpi=600)

plt.plot(N_cache[:,0],N_cache[:,1],color="k",linewidth=3,linestyle='--',alpha=0.4,label='_nolegend_')
d_tar =  plt.scatter(D_target[:,0],D_target[:,1],marker="X",s=200,color="b",label="D_target")
plt.scatter(D_target[:,0],D_target[:,1],marker="X",s=500,color="b",label="D_target")
plt.plot(D_target[:,0],D_target[:,1],linewidth=3,color="k",linestyle='--',alpha=0.4,label='_nolegend_')

act_spc, = plt.plot(action_space_1[:,0],action_space_1[:,1],color="orange",linewidth=4,linestyle='-',marker="o",alpha=0.7,label="Action Space")
act_spc1 = plt.Polygon(np.concatenate((action_space_1,D_0.reshape(-1,2))),alpha=0.4,color="orange",label='_nolegend_')
plt.gca().add_patch(act_spc1)
act_spc2 = plt.Polygon(np.concatenate((action_space_2,D_0.reshape(-1,2))),alpha=0.4,color="purple",label='_nolegend_')
plt.gca().add_patch(act_spc2)

plt.plot(action_space_2[:,0],action_space_2[:,1],color="purple",linewidth=4,linestyle='-',marker="o",alpha=0.7,label='_nolegend_')

gred_acc = plt.arrow(D_0[0],D_0[1],greedy_action_1[1,0],greedy_action_1[1,1],width=2,length_includes_head=True,facecolor="k",edgecolor="w",hatch="\\\\",label="Greedy Action")
plt.arrow(D_0[0],D_0[1],greedy_action_2[1,0],greedy_action_2[1,1],width=2,length_includes_head=True,facecolor="k",edgecolor="w",hatch="\\\\",label='_nolegend_')
plt.arrow(greedy_action_1[1,0],greedy_action_1[1,1],greedy_action_2[1,0],greedy_action_2[1,1],width=2,length_includes_head=True,facecolor="k",edgecolor="w",hatch="\\\\",label='_nolegend_')

orac_acc = plt.arrow(D_0[0],D_0[1],oracle_action_1[1,0],oracle_action_1[1,1],width=2,length_includes_head=True,facecolor="w",edgecolor="k",hatch="OO",label="Oracle Action")
plt.arrow(D_0[0],D_0[1],oracle_action_2[1,0],oracle_action_2[1,1],width=2,length_includes_head=True,facecolor="w",edgecolor="k",hatch="OO",label='_nolegend_')
plt.arrow(oracle_action_2[1,0],oracle_action_2[1,1],oracle_action_1[1,0],oracle_action_1[1,1],width=2,length_includes_head=True,facecolor="w",edgecolor="k",hatch="OO",label='_nolegend_')

error_acc, = plt.plot((total_greedy_action[1,0],total_oracle_action[1,0]),(total_greedy_action[1,1],total_oracle_action[1,1]),color="r",linewidth=3,linestyle='-.',marker="X",label="Error")

#sns.lineplot(x=total_greedy_action[:,0],y=total_greedy_action[:,1],linewidth=3,color="k",linestyle='--',alpha=0.4)
#sns.lineplot(x=total_oracle_action[:,0],y=total_oracle_action[:,1],linewidth=3,color="k",linestyle='--',alpha=0.4)
plt.scatter((total_greedy_action[1,0],total_oracle_action[1,0]),(total_greedy_action[1,1],total_oracle_action[1,1]),marker="X",s=500,color="r",label='_nolegend_')

plt.ylim(np.min(D_target), np.max(D_target)*1.1)
plt.xlim(np.min(D_target), np.max(D_target)*1.1)

plt.ylabel("N images class 2",fontweight="bold" ,fontsize=18)
plt.xlabel("N images class 1",fontweight="bold" ,fontsize=18)

plt.rcParams["font.size"]=15
plt.rcParams["axes.linewidth"]=2
plt.rcParams["legend.labelspacing"] = 0.5
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.legend(handles=[d_tar, act_spc,gred_acc,orac_acc,error_acc], loc='upper left' )
plt.tight_layout()
plt.savefig("toy_example.jpg")