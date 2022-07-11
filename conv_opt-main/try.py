import numpy as np
import cvxpy as cp
from iteround import saferound
n_class = 15
n_device = 20
n_cache = 2
n_size = 200
n_obs = 1000
k=10

D_0 = np.zeros((n_class,1),int)
D_target = np.ones((n_class,1))*n_size/n_class
x_dist = np.random.rand(n_device,n_class)
x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)
N_x = np.zeros((n_device,n_class),dtype=int)
for i in range(n_device):
    N_x[i,:] = saferound(x_dist[i]*n_obs,places=0)
    x_dist[i] = N_x[i,:]
x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)
P_cond = np.random.rand(n_class,n_class)
A_coop = np.zeros((n_device*n_class,1),int)
P_cond = P_cond.reshape(1,n_class,n_class)
P_cond = np.repeat(P_cond,n_device,axis=0)
P_occ = x_dist.reshape(n_device,-1,1) * P_cond
P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)
P_tuple = [P_condr[i] for i in range(n_device)]
P_Condr = np.concatenate(P_tuple,axis=1)

Act = cp.Variable((n_class*n_device,1))

Act_mat = np.concatenate((np.eye(n_class*n_device),-np.eye(n_class*n_device)),axis=0)
eq_mat = np.repeat(np.eye(n_device),n_class,axis=1)
b_eq = np.ones((n_device,1))*n_cache
B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act == k*n_cache ]
obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target))
prob = cp.Problem(obj, constraint)

prob.solve()
y_val = (eq_mat @ Act.value)/n_cache

y_val = np.array(saferound(y_val.reshape(-1),places=0),int)
for i in range(n_device):
    if y_val[i] == 1:
        act = Act.value[i*n_class:(i+1)*n_class]    
        act = act * n_cache/sum(act)
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    else:
        A_coop[i*n_class:(i+1)*n_class] = np.zeros((n_class,1),dtype=int)
print(y_val)
print(sum(A_coop))
print(A_coop)
