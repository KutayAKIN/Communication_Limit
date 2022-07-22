# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import numpy as np
import cvxpy as cp
from iteround import saferound
from scipy.linalg import sqrtm
n_class = 15
n_device = 20
n_cache = 2
n_size = 200
n_obs = 1000
k=10
C = np.zeros((n_device,n_device))
for i in range(n_device):
    for j in range(i):
        C[i,j] = np.random.rand(1,1)
        C[j,i] = C[i,j]

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
y = eq_mat@Act
C = np.transpose(eq_mat)@C@eq_mat
print(C.shape)
C = np.transpose(C)@C
C = sqrtm(C)
print(C.shape)
B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act == k*n_cache ]
obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target)+cp.sum_squares(C@Act))
prob = cp.Problem(obj, constraint)
prob.solve()
print("Is DPP? ", prob.is_dcp(dpp=True))
print("Is DCP? ", prob.is_dcp(dpp=False))