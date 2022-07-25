from re import VERBOSE
from tabnanny import verbose
from tkinter import N
import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from iteround import saferound
import random
import seaborn as sns
import pandas as pd
import os
from torchvision import transforms
from statannot import add_stat_annotation
import sklearn.datasets as ds
import torch
import torchvision.datasets as datasets
from CIFAR10_model import CIFAR10Dataset, ResNet18_10,ResNet18_100
from MNIST_model import MNISTClassifier, MNISTDataset
from AdverseWeather_model import AdverseWeatherDataset
import torchvision.transforms as trfm
import mosek
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
import shutil
import torchvision.models as vsmodels
import json
from PIL import Image
import cv2
import random
from torch.utils.data import DataLoader, Dataset
import argparse

def Cont_Coop_Update_norm(n_device,n_class,D_target,P_Condr,D_0,n_cache,d_size_0):
    A_coop = np.zeros((n_device*n_class,1))
    Act = cp.Variable((n_class*n_device,1))
    Act_mat = np.concatenate((np.eye(n_class*n_device),np.repeat(np.eye(n_device),n_class,axis=1),-np.repeat(np.eye(n_device),n_class,axis=1)),axis=0)
    B = np.concatenate((np.zeros((n_class*n_device,1)),np.ones((n_device,1))*n_cache,-np.ones((n_device,1))*n_cache),axis=0)
    constraint = [Act_mat @ Act - B >= 0]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act -D_target))
    prob = cp.Problem(obj, constraint)

    prob.solve()
    
    A_coop[:] = Act.value

    D_0 = D_0 + np.matmul(P_Condr,A_coop)

    return D_0, A_coop

def Cont_Feed_Update_norm(n_device,n_class,D_target,P_condr,D_0,n_cache,d_size_0,A_ind,n_iter):

    A_feed = np.array(A_ind)
    n=0

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),np.ones((1,n_class)),-np.ones((1,n_class))),axis=0)
            b = np.concatenate((np.zeros((n_class,1)),np.ones((1,1))*n_cache,-np.ones((1,1))*n_cache),axis=0)
            constraint = [act_mat @ act-b >= 0]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_o = A_feed[[j for j in range(n_device*n_class) if j not in k]]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0 + np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act  - D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = act.value
    
    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    D_0 = D_0 + np.matmul(P_Condr,A_feed)

    return D_0,A_feed

def Cont_Ind_Update_norm(n_device,n_class,D_target,P_condr,D_0,n_cache,d_size_0):

    A_ind = np.zeros((n_device*n_class,1))
    for i in range(n_device):
        act = cp.Variable((n_class,1))
        act_mat = np.concatenate((np.eye(n_class),np.ones((1,n_class)),-np.ones((1,n_class))),axis=0)
        b = np.concatenate((np.zeros((n_class,1)),np.ones((1,1))*n_cache,-np.ones((1,1))*n_cache),axis=0)
        constraint = [act_mat @ act-b >= 0]
        obj = cp.Minimize(cp.sum_squares(D_0+ P_condr[i] @ act-D_target))
        prob = cp.Problem(obj, constraint)
        prob.solve()
        A_ind[i*n_class:(i+1)*n_class] = act.value

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    D_0 = D_0+ np.matmul(P_Condr,A_ind)

    return D_0, A_ind

def Cont_Unif_Update_norm(n_device,n_class,P_condr,D_0,n_cache,d_size_0):

    A_feed = np.ones((n_device*n_class,1))*(n_cache/n_class)

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    D_0 = D_0 + np.matmul(P_Condr,A_feed)

    return D_0,A_feed

def Cont_Lwb_Update_norm(n_device,n_class,D_target,P_Condr,D_0,n_cache,d_size_0):
    A_coop = np.zeros((n_device*n_class,1))
    A = np.eye(n_class).reshape(1,n_class,n_class).repeat(n_device,axis=0)
    A_tuple = [A[i] for i in range(n_device)]
    A = np.concatenate(A_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    
    Act_mat = np.concatenate((np.eye(n_class*n_device),np.repeat(np.eye(n_device),n_class,axis=1),-np.repeat(np.eye(n_device),n_class,axis=1)),axis=0)
    
    B = np.concatenate((np.zeros((n_class*n_device,1)),np.ones((n_device,1))*n_cache,-np.ones((n_device,1))*n_cache),axis=0)
    constraint = [Act_mat @ Act - B >= 0]
    
    
    obj = cp.Minimize(cp.sum_squares(D_0 + A @ Act -D_target))
    prob = cp.Problem(obj, constraint)

    prob.solve()
    
    A_coop[:] = Act.value

    D_0 = D_0 + np.matmul(A,A_coop)

    return D_0, A_coop

def Int_Unif_Update(n_device,n_class,D_0,n_cache,d_size_0,Xs,X_samples,N_x):

    A_feed = np.zeros((n_device*n_class,1),int)

    for i in range(n_device):
        n_lim = n_cache
        args = [i for i in range(n_class)]
        random.shuffle(args)
        fulls = []
        while n_lim !=0:
            rem_len = n_class - len(fulls)
            n_cache_cand = saferound(np.ones(rem_len)*n_lim/rem_len,places=0)
            f = [arg for arg in args if (arg not in fulls)]
            for g,arg in enumerate(f): 
                if N_x[i][arg]<=n_cache_cand[g]:
                    A_feed[i*n_class+arg] += int(N_x[i][arg])
                    fulls.append(arg)
                    n_lim = n_lim - int(N_x[i][arg])
                else:
                    A_feed[i*n_class+arg] += int(n_cache_cand[g])
                    n_lim = n_lim - int(n_cache_cand[g])
        
    for i in range(n_device):
        n_cached_ind = A_feed[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(X_samples[i] == j).tolist(), k = n_cached_ind[j][0])
            for k in ind_samples:
                D_0[Xs[i,k]] +=1

    return D_0,A_feed

def Int_Ind_Update_norm(n_device,n_class,D_target,P_condr,D_0,n_cache,d_size_0,Xs,X_samples,N_x):

    A_ind = np.zeros((n_device*n_class,1),int)

    for i in range(n_device):
        act = cp.Variable((n_class,1))
        act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
        eq_mat = np.ones((1,n_class))
        b_eq = np.ones((1,1))*n_cache
        b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
        constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]
        
        obj = cp.Minimize(cp.sum_squares(D_0+ P_condr[i] @ act -D_target))
        prob = cp.Problem(obj, constraint)
        prob.solve()
        A_ind[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    D_0 = np.array(D_0,int)

    for i in range(n_device):
        n_cached_ind = A_ind[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(X_samples[i] == j).tolist(), k = n_cached_ind[j][0])
            for k in ind_samples:
                D_0[Xs[i,k]] +=1

    return D_0, A_ind

def Int_Coop_Update_norm(n_device,n_class,D_target,P_Condr,D_0,n_cache,d_size_0,Xs,X_samples,N_x):
    A_coop = np.zeros((n_device*n_class,1),int)
    Act = cp.Variable((n_class*n_device,1))

    Act_mat = np.concatenate((np.eye(n_class*n_device),-np.eye(n_class*n_device)),axis=0)
    eq_mat = np.repeat(np.eye(n_device),n_class,axis=1)
    b_eq = np.ones((n_device,1))*n_cache
    B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
    constraint = [Act_mat @ Act >= B, eq_mat @ Act == b_eq]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target))
    prob = cp.Problem(obj, constraint)

    prob.solve()
    for i in range(n_device):
        act = Act.value[i*n_class:(i+1)*n_class]
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)

    D_0 = np.array(D_0,int)

    for i in range(n_device):
        n_cached_ind = A_coop[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(X_samples[i] == j).tolist(), k = n_cached_ind[j][0])
            for k in ind_samples:
                D_0[Xs[i,k]] +=1
    
    return D_0, A_coop

def Int_Feed_Update_norm(n_device,n_class,D_target,P_condr,D_0,n_cache,d_size_0,A_ind,n_iter,Xs,X_samples,N_x):

    A_feed = np.array(A_ind,int)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_o = A_feed[[j for j in range(n_device*n_class) if j not in k]]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    for i in range(n_device):
        n_cached_ind = A_feed[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(X_samples[i] == j).tolist(), k = n_cached_ind[j][0])
            for k in ind_samples:
                D_0[Xs[i,k]] +=1

    return D_0,A_feed

def Int_Lwb_Update_norm(n_device,n_class,D_target,P_Condr,D_0,n_cache,d_size_0):
    A_coop = np.zeros((n_device*n_class,1),int)
    A = np.eye(n_class).reshape(1,n_class,n_class).repeat(n_device,axis=0)
    A_tuple = [A[i] for i in range(n_device)]
    A = np.concatenate(A_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    
    Act_mat = np.concatenate((np.eye(n_class*n_device),np.repeat(np.eye(n_device),n_class,axis=1),-np.repeat(np.eye(n_device),n_class,axis=1)),axis=0)
    
    B = np.concatenate((np.zeros((n_class*n_device,1)),np.ones((n_device,1))*n_cache,-np.ones((n_device,1))*n_cache),axis=0)
    constraint = [Act_mat @ Act - B >= 0]
    
    
    obj = cp.Minimize(cp.sum_squares(D_0 + A @ Act -D_target))
    prob = cp.Problem(obj, constraint)

    prob.solve()

    for i in range(n_device):
        act = Act.value[i*n_class:(i+1)*n_class]
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)

    D_0 = D_0 + np.matmul(A,A_coop)
    
    return D_0, A_coop

def Int_Unif_Action(n_device,n_class,n_cache,N_x):

    A_feed = np.zeros((n_device*n_class,1),int)

    for i in range(n_device):
        n_lim = n_cache
        args = [i for i in range(n_class)]
        random.shuffle(args)
        fulls = []
        while n_lim !=0:
            rem_len = n_class - len(fulls)
            n_cache_cand = saferound(np.ones(rem_len)*n_lim/rem_len,places=0)
            f = [arg for arg in args if (arg not in fulls)]
            for g,arg in enumerate(f): 
                if N_x[i][arg]<=n_cache_cand[g]:
                    A_feed[i*n_class+arg] += int(N_x[i][arg])
                    fulls.append(arg)
                    n_lim = n_lim - int(N_x[i][arg])
                else:
                    A_feed[i*n_class+arg] += int(n_cache_cand[g])
                    n_lim = n_lim - int(n_cache_cand[g])
    return A_feed

def Int_Lwb_Action_norm(n_device,n_class,D_target,D_0,n_cache):

    A_coop = np.zeros((n_device*n_class,1))
    A = np.eye(n_class).reshape(1,n_class,n_class).repeat(n_device,axis=0)
    A_tuple = [A[i] for i in range(n_device)]
    A = np.concatenate(A_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    
    Act_mat = np.concatenate((np.eye(n_class*n_device),np.repeat(np.eye(n_device),n_class,axis=1),-np.repeat(np.eye(n_device),n_class,axis=1)),axis=0)
    
    B = np.concatenate((np.zeros((n_class*n_device,1)),np.ones((n_device,1))*n_cache,-np.ones((n_device,1))*n_cache),axis=0)
    constraint = [Act_mat @ Act - B >= 0]
    
    obj = cp.Minimize(cp.sum_squares(D_0 + A @ Act -D_target))
    prob = cp.Problem(obj, constraint)

    prob.solve()
    A_coop[:] = Act.value
    A_m = np.array(saferound(np.matmul(A,A_coop).reshape(-1),places=0),int).reshape(-1,1)
    D_0 = D_0 + A_m
    
    return D_0

def Int_Coop_Action_norm(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x):
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
    constraint = [Act_mat @ Act >= B, eq_mat @ Act == b_eq]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target))
    prob = cp.Problem(obj, constraint)

    prob.solve()

    for i in range(n_device):
        act = Act.value[i*n_class:(i+1)*n_class]
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    
    return A_coop

def Int_Coop_Action_norm_group(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x, k):
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
    constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act <= k*n_cache ]
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
    
    return A_coop

def Int_Coop_Action_norm_group2(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x, C, k):
    A_coop = np.zeros((n_device*n_class,1),int)
    P_cond = P_cond.reshape(1,n_class,n_class)
    alpha = 0.1
    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    Act2 = cp.Variable((n_class*n_device,1))
    Act_mat = np.concatenate((np.eye(n_class*n_device),-np.eye(n_class*n_device)),axis=0)
    eq_mat = np.repeat(np.eye(n_device),n_class,axis=1)
    b_eq = np.ones((n_device,1))*n_cache
    B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
    y = eq_mat@Act
    constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act <= k*n_cache ]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target)+C@y)
    prob = cp.Problem(obj, constraint)
    prob.solve()
    y_val = (eq_mat @ Act.value)/n_cache
    y_val = np.array(saferound(y_val.reshape(-1),places=0),int)
    for d in range(n_device):
        if y_val[d] == 0:
            B[n_device*n_class+d*n_class:(d+1)*n_class+(n_device*n_class)] = np.zeros((n_class,1))

    b_eq= b_eq*n_device/k
    constraint = [Act_mat @ Act2 >= B, eq_mat @ Act2 <= b_eq]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act2 - D_target))
    prob = cp.Problem(obj, constraint)
    prob.solve()
    for i in range(n_device):
        act = Act2.value[i*n_class:(i+1)*n_class]    
        #act = act * n_cache/sum(act)
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    
    return A_coop, y_val

def Int_Coop_Action_norm_group3(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x, C, k):
    A_coop = np.zeros((n_device*n_class,1),int)
    P_cond = P_cond.reshape(1,n_class,n_class)
    alpha = 0.1
    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    Act2 = cp.Variable((n_class*n_device,1))
    Act_mat = np.concatenate((np.eye(n_class*n_device),-np.eye(n_class*n_device)),axis=0)
    eq_mat = np.repeat(np.eye(n_device),n_class,axis=1)
    b_eq = np.ones((n_device,1))*n_cache
    B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
    y = eq_mat@Act
    constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act <= k*n_cache ]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target)+C@y)
    prob = cp.Problem(obj, constraint)
    prob.solve()
    y_val = np.random.permutation(np.concatenate((np.ones((k,1),int),np.zeros((n_device-k,1),int)),axis=0))
    #y_val = np.array(saferound(y_val.reshape(-1),places=0),int)
    for d in range(n_device):
        if y_val[d] == 0:
            B[n_device*n_class+d*n_class:(d+1)*n_class+(n_device*n_class)] = np.zeros((n_class,1))
    #B= B*n_device/k
    b_eq = b_eq * n_device/k
    constraint = [Act_mat @ Act2 >= B, eq_mat @ Act2 <= b_eq]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act2 - D_target))
    prob = cp.Problem(obj, constraint)
    prob.solve()
    for i in range(n_device):
        act = Act2.value[i*n_class:(i+1)*n_class]    
        #act = act * n_cache/sum(act)
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    
    return A_coop, y_val

def Int_Coop_Action_norm_group4(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x, C, k):
    A_coop = np.zeros((n_device*n_class,1),int)
    P_cond = P_cond.reshape(1,n_class,n_class)
    alpha = 0.1
    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    Act2 = cp.Variable((n_class*n_device,1))
    Act_mat = np.concatenate((np.eye(n_class*n_device),-np.eye(n_class*n_device)),axis=0)
    eq_mat = np.repeat(np.eye(n_device),n_class,axis=1)
    b_eq = np.ones((n_device,1))*n_cache
    B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
    y = eq_mat@Act
    constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act <= k*n_cache ]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target)+0.1*C@y)
    prob = cp.Problem(obj, constraint)
    prob.solve()
    y_val = np.random.permutation(np.concatenate((np.ones((k,1),int),np.zeros((n_device-k,1),int)),axis=0))
    #y_val = np.array(saferound(y_val.reshape(-1),places=0),int)
    for d in range(n_device):
        if y_val[d] == 0:
            B[n_device*n_class+d*n_class:(d+1)*n_class+(n_device*n_class)] = np.zeros((n_class,1))
    #B= B*n_device/k
    b_eq = b_eq * n_device/k
    constraint = [Act_mat @ Act2 >= B, eq_mat @ Act2 <= b_eq]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act2 - D_target))
    prob = cp.Problem(obj, constraint)
    prob.solve()
    for i in range(n_device):
        act = Act2.value[i*n_class:(i+1)*n_class]    
        #act = act * n_cache/sum(act)
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    
    return A_coop, y_val

def Int_Coop_Action_norm_group5(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x, C, k):
    A_coop = np.zeros((n_device*n_class,1),int)
    P_cond = P_cond.reshape(1,n_class,n_class)
    alpha = 0.1
    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    Act = cp.Variable((n_class*n_device,1))
    Act2 = cp.Variable((n_class*n_device,1))
    Act_mat = np.concatenate((np.eye(n_class*n_device),-np.eye(n_class*n_device)),axis=0)
    eq_mat = np.repeat(np.eye(n_device),n_class,axis=1)
    b_eq = np.ones((n_device,1))*n_cache
    B = np.concatenate((np.zeros((n_class*n_device,1)),-N_x.reshape(-1,1)),axis=0)
    y = eq_mat@Act
    constraint = [Act_mat @ Act >= B, eq_mat @ Act <= b_eq, np.ones((1,n_device)) @ eq_mat @ Act <= k*n_cache ]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act - D_target)+0.1*C@y)
    prob = cp.Problem(obj, constraint)
    prob.solve()
    y_val = np.random.permutation(np.concatenate((np.ones((k,1),int),np.zeros((n_device-k,1),int)),axis=0))
    #y_val = np.array(saferound(y_val.reshape(-1),places=0),int)
    for d in range(n_device):
        if y_val[d] == 0:
            B[n_device*n_class+d*n_class:(d+1)*n_class+(n_device*n_class)] = np.zeros((n_class,1))
    #B= B*n_device/k
    b_eq = b_eq * n_device/k
    constraint = [Act_mat @ Act2 >= B, eq_mat @ Act2 <= b_eq]
    obj = cp.Minimize(cp.sum_squares(D_0 + P_Condr @ Act2 - D_target))
    prob = cp.Problem(obj, constraint)
    prob.solve()
    for i in range(n_device):
        act = Act2.value[i*n_class:(i+1)*n_class]    
        #act = act * n_cache/sum(act)
        A_coop[i*n_class:(i+1)*n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    
    return A_coop, y_val

def Int_Feed_Action_norm_cluster(n_device,n_class,D_target,P_cond,D_0,n_cache,A_ind,n_iter,x_dist,N_x, counts):

    A_feed = np.array(A_ind,int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache*counts[i]
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_o = A_feed[[j for j in range(n_device*n_class) if j not in k]]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1).squeeze()

    return A_feed

def Int_Feed_Action_norm(n_device,n_class,D_target,P_cond,D_0,n_cache,A_ind,n_iter,x_dist,N_x):

    A_feed = np.array(A_ind,int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_o = A_feed[[j for j in range(n_device*n_class) if j not in k]]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    return A_feed

def Int_Feed_Action_norm_my10_aftercluster(n_device,n_class,A_cluster,P_cond,n_cache,A_ind,n_iter,x_dist,N_x,a_m, labels):

    A_feed = np.array(A_ind,int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_s = A_feed[[j for j in range(n_device*n_class) if j not in k]]
            a_m_p = np.zeros((n_device, n_device-1))
            for l in range(n_device):
                k = 0
                for j in range(n_device):
                    if j == i:
                        continue
                    else:
                        a_m_p[l][k] = a_m[l][j]
                        k = k + 1
            indices = a_m_p[i]

            A_feed_o = np.zeros(((n_device-1)*n_class,1))
            for d in range(n_device-1):
                if indices[d] == 1:
                    for p in range(d*n_class, (d+1)*n_class):
                        A_feed_o[p] = A_feed_s[p]

        #A_feed_o = np.zeros(A_feed.shape) + A_feed[indices]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- A_cluster[labels[i]]))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1).ravel()

    return A_feed

def Int_Feed_Action_norm_my10(n_device,n_class,D_target,P_cond,D_0,n_cache,A_ind,n_iter,x_dist,N_x,a_m):

    A_feed = np.array(A_ind,int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_s = A_feed[[j for j in range(n_device*n_class) if j not in k]]
            a_m_p = np.zeros((n_device, n_device-1))
            for l in range(n_device):
                k = 0
                for j in range(n_device):
                    if j == i:
                        continue
                    else:
                        a_m_p[l][k] = a_m[l][j]
                        k = k + 1
            indices = a_m_p[i]

            A_feed_o = np.zeros(((n_device-1)*n_class,1))
            for d in range(n_device-1):
                if indices[d] == 1:
                    for p in range(d*n_class, (d+1)*n_class):
                        A_feed_o[p] = A_feed_s[p]

        #A_feed_o = np.zeros(A_feed.shape) + A_feed[indices]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    return A_feed

def Int_Feed_Action_norm_my25(n_device,n_class,D_target,P_cond,D_0,n_cache,A_ind,n_iter,x_dist,N_x,a_m):

    A_feed = np.array(A_ind,int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_s = A_feed[[j for j in range(n_device*n_class) if j not in k]]
            obtions = [x for x in random.sample(range(0,n_device-1),4)]
            obtions = 15*obtions
            indices = [j for j in np.concatenate([np.arange(obtions[0], obtions[0]+15), np.arange(obtions[1], obtions[1]+15), np.arange(obtions[2], obtions[2]+15) , np.arange(obtions[3], obtions[3]+15)])]
            #A_feed_o = A_feed[indices]
            A_feed_o = np.zeros((2985,1))
            for d in range((n_device-1)*n_class):
                if d in indices:
                    A_feed_o[d] = A_feed_s[d]

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    return A_feed

def Int_Feed_Action_norm_my50(n_device,n_class,D_target,P_cond,D_0,n_cache,A_ind,n_iter,x_dist,N_x,a_m):

    A_feed = np.array(A_ind,int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for _ in range(n_iter):
        for i in range(n_device):
            act = cp.Variable((n_class,1))
            act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
            eq_mat = np.ones((1,n_class))
            b_eq = np.ones((1,1))*n_cache
            b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

            k = [l for l in range(i*n_class,(i+1)*n_class)]
            A_feed_s = A_feed[[j for j in range(n_device*n_class) if j not in k]]
            
            obtions = [x for x in random.sample(range(0,n_device-1),20)]
            obtions = 15*obtions
            indices = [j for j in np.concatenate([np.arange(obtions[0], obtions[0]+15), np.arange(obtions[1], obtions[1]+15), np.arange(obtions[2], obtions[2]+15) , np.arange(obtions[3], obtions[3]+15),
             np.arange(obtions[4], obtions[4]+15),  np.arange(obtions[5], obtions[5]+15),  np.arange(obtions[6], obtions[6]+15),  np.arange(obtions[7], obtions[7]+15) ,  np.arange(obtions[8], obtions[8]+15),
             np.arange(obtions[9], obtions[9]+15), np.arange(obtions[10], obtions[10]+15) , np.arange(obtions[11], obtions[11]+15), np.arange(obtions[12], obtions[12]+15), np.arange(obtions[13], obtions[13]+15), 
             np.arange(obtions[14], obtions[14]+15), np.arange(obtions[15], obtions[15]+15), np.arange(obtions[16], obtions[16]+15), np.arange(obtions[17], obtions[17]+15), np.arange(obtions[18], obtions[18]+15),
             np.arange(obtions[19], obtions[19]+15)])]

            #A_feed_o = A_feed[indices]
            A_feed_o = np.zeros((2985,1))
            for d in range((n_device-1)*n_class):
                if d in indices:
                    A_feed_o[d] = A_feed_s[d]
               

            P_tuple = [P_condr[l] for l in range(n_device) if l!=i]
            P_Condr_o = np.concatenate(P_tuple,axis=1)

            obj = cp.Minimize(cp.sum_squares(D_0+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
    
            prob = cp.Problem(obj,constraint)
            prob.solve()

            A_feed[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    return A_feed

def suggested_C_mat(n_device, n_class):
    C_mat = np.zeros((n_class,n_class,n_device))
    for i in range(n_device):
        for j in range(n_class):
            for k in range(n_class):
                if j != k:
                    C_mat[k,j,i] = np.absolute(np.random.normal(0,1,1))
                else:
                    continue
    A = np.sum(C_mat, axis=0)
    for i in range(n_device):
        for j in range(n_class):
            for k in range(n_class):
                if j == k:
                    while C_mat[k,j,i] < A[j,i]:
                        C_mat[k,j,i] = np.absolute(np.random.randint(np.ceil(A[j,i])-2,high = 5*np.ceil(A[j,i]), size = 1))
                else:
                    continue
    B = np.sum(C_mat, axis=0)
    for i in range(n_device):
        for j in range(n_class):
            for k in range(n_class):
                C_mat[k,j,i] = C_mat[k,j,i] / B[j,i]

    best_selections = np.zeros((n_device-1,n_device), dtype=np.int8)
    for i in range(n_device):
        used = np.array([i])
        base = C_mat[:,:,i]
        besti = np.empty([n_device-1], dtype=np.int8)
        for k in range(n_device-1):
            
            max_score = np.array([0])
            for j in range(n_device):
                if j in used:
                    continue
                else:
                    mat = base + C_mat[:,:,j]
                    score = np.linalg.det(np.transpose(mat))/np.math.factorial(n_class)
                    if score > max_score:
                        max_mat = C_mat[:,:,j]
                        max_score = score
                        index = j
            used = np.append(used, index)
            besti[k] = index
            base = base + C_mat[:,:,index]
        best_selections[:,i] = besti
    return C_mat ,best_selections

def Int_Ind_Action_norm(n_device,n_class,D_target,P_cond,D_0,n_cache,x_dist,N_x,k):

    A_ind = np.zeros((n_device*n_class,1),int)

    P_cond = P_cond.reshape(1,n_class,n_class)

    P_cond = np.repeat(P_cond,n_device,axis=0)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    for i in range(n_device):
        act = cp.Variable((n_class,1))
        act_mat = np.concatenate((np.eye(n_class),-np.eye(n_class)),axis=0)
        eq_mat = np.ones((1,n_class))
        b_eq = np.ones((1,1))*n_cache*n_device/k
        b = np.concatenate((np.zeros((n_class,1)),-N_x[i,:].reshape(-1,1)),axis=0)
        constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]
        
        obj = cp.Minimize(cp.sum_squares(D_0+ P_condr[i] @ act -D_target))
        prob = cp.Problem(obj, constraint)
        prob.solve()
        A_ind[i*n_class:(i+1)*n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

    return A_ind

def generate_Ys_Y_hat(P_cond,n_device,n_class,n_obs,N_x):
    c = [i for i in range(n_class)]
    X_samples = np.zeros((n_device,n_obs),dtype=int)
    Xs = np.zeros((n_device,n_obs),dtype=int)
    for i in range(n_device):
        ind= 0
        for j in range(n_class):
            cond_prob = P_cond[i,j]
            Xs[i,ind:ind+N_x[i,j]] = j
            X_samples[i,ind:ind+N_x[i,j]] = random.choices(c,weights=cond_prob,k=N_x[i,j])
            ind += N_x[i,j]
    

    N_x_hat = np.zeros_like(N_x,int)

    for i in range(n_device):
        for j in range(n_class):
            N_x_hat[i,j] = sum(X_samples[i,:] == j)

    return Xs,X_samples, N_x_hat

def generate_D0(n_class,d_size_0):
    D_0 = np.random.rand(n_class,1)
    #D_0 = np.ones((n_class,1))
    #D_0[3] = 0
    #D_0[4] = 0
    #D_0[5] = 0
    #D_0[6] = 0
    D_0 = D_0 / np.sum(D_0)
    D_0 = saferound(D_0[:,0]*d_size_0,places=0)
    D_0_ind = np.array(D_0,dtype=int).reshape(-1,1)
    D_0_coop = np.array(D_0,dtype=int).reshape(-1,1)
    D_0_feed = np.array(D_0,dtype=int).reshape(-1,1)
    return D_0_ind,D_0_coop,D_0_feed

def generate_xdist_Nx(n_device,n_class,n_obs):
    x_dist = np.random.rand(n_device,n_class)
    #x_dist = np.random.rand(1,n_class)
    #x_dist = x_dist.repeat(n_device,axis=0)

    #x_dist = np.array([[1,1,1,1,1,0,0],[1,1,1,1,0,1,1],[1,1,1,1,0,1,0]])
    x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    N_x = np.zeros((n_device,n_class),dtype=int)
    for i in range(n_device):
        N_x[i,:] = saferound(x_dist[i]*n_obs,places=0)
        x_dist[i] = N_x[i,:]
    x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    return x_dist,N_x

def generate_cond_probs(n_class,n_device,x_dist,acc):
    P_cond = np.random.rand(n_device,n_class,n_class)
    for i in range(n_device):
        for j in range(n_class):
            sumk = np.sum(P_cond[i,j,:]) - P_cond[i,j,j]
            P_cond[i,j,:] *= (1-acc)/sumk
            P_cond[i,j,j] = acc
    
    P_cond = P_cond /np.sum(P_cond,axis=2).reshape(n_device,-1,1)

    P_occ = x_dist.reshape(n_device,-1,1) * P_cond

    P_condr = P_occ /(np.sum(P_occ,1).reshape(n_device,1,-1)+1e-10)

    P_tuple = [P_condr[i] for i in range(n_device)]
    P_Condr = np.concatenate(P_tuple,axis=1)

    return P_cond, P_condr, P_Condr

def create_synt_dataset(save_dir, num_features,num_class, num_sample, train_ratio,val_ratio):
    X, y = ds.make_blobs(n_samples=num_sample * num_class, centers=num_class, n_features=num_features, random_state=10)
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    X_diff = X_max - X_min
    Xn = 2 * ((X - X_min) / X_diff - 0.5)

    # Split it into train and test set

    X_train, X_test, y_train, y_test = train_test_split(Xn, y, train_size=train_ratio, stratify=y,random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, stratify=y_train,random_state=1)

    isExist = os.path.exists(save_dir)

    if not isExist:
        os.makedirs(save_dir)

    np.save(save_dir + '/X_train.npy', X_train)
    np.save(save_dir + '/X_test.npy', X_test)
    np.save(save_dir + '/X_val.npy', X_val)
    np.save(save_dir + '/y_train.npy', y_train)
    np.save(save_dir + '/y_test.npy', y_test)
    np.save(save_dir + '/y_val.npy', y_val)

def load_synt_dataset(dir, show_plots=False):
    X_train = np.load(dir + '/X_train.npy')
    X_test = np.load(dir + '/X_test.npy')
    X_val = np.load(dir+"/X_val.npy")
    y_train = np.load(dir + '/y_train.npy')
    y_test = np.load(dir + '/y_test.npy')
    y_val = np.load(dir + '/y_val.npy')

    return X_train, X_test, X_val, y_train, y_test, y_val

def create_dataset(X, y,n_features):
    data = torch.tensor(X, dtype=torch.float)
    label = torch.tensor(y, dtype=int)
    dataset = torch.zeros((data.shape[0], n_features+1))
    dataset[:, :n_features] = data
    dataset[:, n_features] = label
    return dataset

def create_base_dataset(X,y,base_class,d_size,n_features):
    dataset = torch.zeros((d_size,n_features+1))

    x_dist = np.random.rand(len(base_class))

    x_dist = x_dist / sum(x_dist)

    N_x = list(map(int,saferound(x_dist*d_size,places=0)))
    ind = 0
    for i,c_i in enumerate(base_class):
        ind_s = random.sample(np.argwhere(y == c_i).tolist(), k = N_x[i])
        dataset[ind:ind+N_x[i],:n_features] = torch.tensor(X[tuple(ind_s),:]).reshape(-1,n_features)
        dataset[ind:ind+N_x[i],n_features] = c_i
        ind += N_x[i] 
    return dataset

def create_MNIST_base_dataset(X,y,base_class,size):
    Xs,ys = X[:size].clone().detach(),y[:size].clone().detach()
    x_dist = np.random.rand(len(base_class))
    x_dist = x_dist / sum(x_dist)
    N_x = list(map(int,saferound(x_dist*size,places=0)))
    ind = 0
    for i,c_i in enumerate(base_class):
        ind_s = random.sample(np.argwhere(y == c_i)[0].tolist(), k = N_x[i])
        Xs[ind:ind+N_x[i],:] = X[ind_s,:].clone().detach()
        ys[ind:ind+N_x[i]] = y[ind_s].clone().detach()
        ind += N_x[i]
    dataset = create_MNIST_dataset(Xs,ys)
    return dataset

def create_CIFAR_base_dataset(X,y,base_class,size):
    data_inds = list()
    Xs,ys = X[:size].clone().detach(),y[:size].clone().detach()
    x_dist = np.random.rand(len(base_class))
    x_dist = x_dist / sum(x_dist)
    N_x = list(map(int,saferound(x_dist*size,places=0)))
    ind = 0
    for i,c_i in enumerate(base_class):
        ind_s = random.sample(np.argwhere(y == c_i)[0].tolist(), k = N_x[i])
        Xs[ind:ind+N_x[i],:] = X[ind_s,:].clone().detach()
        ys[ind:ind+N_x[i]] = y[ind_s].clone().detach()
        ind += N_x[i]
        data_inds.append(ind_s)
    dataset = create_CIFAR10_traindataset(Xs,ys)
    return dataset, data_inds

def dataset_MNIST_stat(dataset,n_class):
    N_x = np.zeros((n_class,1),int)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    for _,y in test_loader:
        for i in range(n_class):
            N_x[i] += sum(y==i).item()
    return N_x

def dataset_stat(dataset,n_class,n_features):
    N_x = np.zeros((n_class,1),int)
    for i in range(n_class):
        N_x[i] = sum(dataset[:,n_features] == i).item()
    return N_x

def cond_preb_calc(model,test_dataset,device, n_class,n_features):
    conf_matrix = test_function(model,test_dataset, n_class,n_features,device)
    cond_prob = conf_matrix /(np.sum(conf_matrix,1).reshape(1,-1)+1e-10)

    return cond_prob

def cond_preb_calc_MNIST(model,test_dataset,device):
    conf_matrix = test_MNIST_function(model,test_dataset,device)
    cond_prob = conf_matrix /(np.sum(conf_matrix,1).reshape(1,-1)+1e-10)


    return cond_prob

def cond_preb_calc_CIFAR(model,test_dataset,device):
    conf_matrix = test_CIFAR_function(model,test_dataset,device,100)
    cond_prob = conf_matrix /(np.sum(conf_matrix,1).reshape(1,-1)+1e-10)

    return cond_prob

def create_dataloader(dataset, b_size, device,n_features):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0)
    data = torch.zeros((b_size,n_features), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    return dataloader, data, label

def train_model(model, losses, loss_fn, optimizer, num_epoch, data, label,n_features, dataloader, silent=True):
    model.train()

    if not silent:
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)
    else:
        pbar = [i for i in range(num_epoch)]
    for epoch in pbar:
        for data_i in dataloader:
            data[:, :] = data_i[:, :n_features]
            label[:] = data_i[:, n_features]

            model.zero_grad()

            out = model(data)
            loss = loss_fn(out, label)
            losses.append(loss)

            loss.backward()

            optimizer.step()
        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    return losses

def test_function(model, test_dataset, n_class,n_features,device):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    data = torch.zeros((len(test_dataset), n_features), dtype=torch.float).to(device)
    label = torch.zeros((len(test_dataset)), dtype=int).to(device)
    confusion_matrix = np.zeros((n_class, n_class), dtype=int)
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:

            data[:, :] = test_data[:, :n_features]
            label[:] = test_data[:, n_features]

            out = model(data)

            _, pred = torch.max(out.data, 1)

            for i in range(n_class):
                filt_i = (label == i)
                pred_i = pred[filt_i]
                for j in range(n_class):
                    filt_j = (pred_i == j)
                    nnum = sum(filt_j)
                    confusion_matrix[i, j] += nnum

    return confusion_matrix

def acc_calc(confusion_matrix):
    return np.trace(confusion_matrix)/np.sum(confusion_matrix)

def create_xdist(n_device,n_class,obs_clss,n_obs):
    x_dist = np.zeros((n_device,n_class))
    for i in range(n_device):
        for j in obs_clss[i]:
            x_dist[i,j] = np.random.rand()
    
    x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    N_x = np.zeros((n_device,n_class),dtype=int)
    for i in range(n_device):
        N_x[i,:] = saferound(x_dist[i]*n_obs,places=0)
        x_dist[i] = N_x[i,:]
    x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    return x_dist, N_x

def create_xdist2(n_device,n_class,obs_clss,n_obs):
    x_dist = np.zeros((n_device,n_class))
    for i in range(n_device):
        for j in obs_clss[i]:
            x_dist[i,j] = np.random.rand()
    for i in range(n_device):
        x_dist[i,1]  =  np.ceil(np.sum(x_dist[i],0)) + np.random.rand()
        x_dist[i] = np.random.permutation(x_dist[i])

    x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    N_x = np.zeros((n_device,n_class),dtype=int)
    for i in range(n_device):
        N_x[i,:] = saferound(x_dist[i]*n_obs,places=0)
        x_dist[i] = N_x[i,:]
    x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

    return x_dist, N_x

def create_obs_datasets(N_x,n_obs,X,y,n_features):
    dataset = torch.zeros((n_obs,n_features+1))
    ind = 0
    for i in range(len(N_x)):
        ind_s = random.sample(np.argwhere(y == i).tolist(), k = N_x[i])
        dataset[ind:ind+N_x[i],:n_features] = torch.tensor(X[ind_s,:]).reshape(-1,n_features)
        dataset[ind:ind+N_x[i],n_features] = i
        ind += N_x[i]
    return dataset

def create_MNIST_obs_datasets(N_x,n_obs,X,y):
    Xs = X[:n_obs].clone().detach()
    ys = y[:n_obs].clone().detach()
    ind = 0
    for i in range(len(N_x)):
        ind_s = random.sample(np.argwhere(y == i)[0].tolist(), k = N_x[i])
        Xs[ind:ind+N_x[i],:] = X[ind_s].clone().detach()
        ys[ind:ind+N_x[i]] = y[ind_s].clone().detach() 
        ind += N_x[i]
    return create_MNIST_dataset(Xs,ys)

def create_CIFAR_obs_datasets(N_x,n_obs,X,y):
    Xs = X[:n_obs].clone().detach()
    ys = y[:n_obs].clone().detach()
    ind = 0
    for i in range(len(N_x)):
        ind_s = random.sample(np.argwhere(y == i)[0].tolist(), k = N_x[i])
        Xs[ind:ind+N_x[i],:] = X[ind_s].clone().detach()
        ys[ind:ind+N_x[i]] = y[ind_s].clone().detach() 
        ind += N_x[i]
    return create_CIFAR10_dataset(Xs,ys)

def eval_obs(model,obs_dataset,device,n_features):
    
    model.eval()
    y_pred = np.zeros((len(obs_dataset)),int)
    data = torch.zeros((len(obs_dataset), n_features), dtype=torch.float).to(device)
    with torch.no_grad():
        data[:,:] = obs_dataset[:,:n_features]

        out = model(data)

        _, pred = torch.max(out.data, 1)

        y_pred[:] = pred.cpu().numpy()
    return y_pred

def eval_MNIST_obs(model, obs_dataset, device):
    test_loader = torch.utils.data.DataLoader(obs_dataset, batch_size=len(obs_dataset))
    data = torch.zeros((len(obs_dataset), 1, 28, 28), dtype=torch.float).to(device)
    y_pred = np.zeros(len(obs_dataset),int)
    model.eval()
    i =0
    with torch.no_grad():
        for x,_ in test_loader:

            data[:] = x.reshape(data.shape)

            out = model(data)

            _, pred = torch.max(out.data, 1)

            y_pred[:] = pred.cpu().numpy()
    return y_pred

def eval_CIFAR_obs(model, obs_dataset, device):
    test_loader = torch.utils.data.DataLoader(obs_dataset, batch_size=len(obs_dataset))
    data = torch.zeros((len(obs_dataset), 3, 32, 32), dtype=torch.float).to(device)
    y_pred = np.zeros(len(obs_dataset),int)
    model.eval()
    i =0
    with torch.no_grad():
        for x,_ in test_loader:

            data[:] = x.reshape(data.shape)

            out = model(data)

            _, pred = torch.max(out.data, 1)

            y_pred[:] = pred.cpu().numpy()
    return y_pred

def y_pred_stats(y_pred,n_class):
    N_x = np.zeros((n_class,1),int)
    for i in range(n_class):
        N_x[i] = np.sum(y_pred == i)
    return N_x

def cache_imgs_feed(A,obs_dataset,y_pred,n_cache,n_device,n_class,n_features,k_num):
    cached_dataset = torch.zeros((n_device*n_cache,n_features+1))
    ind = 0
    for i in range(n_device):
        n_cached_ind = A[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(y_pred[i]==j).tolist(), k= n_cached_ind[j][0])  
            cached_dataset[ind:ind+n_cached_ind[j][0],:] = obs_dataset[i][ind_samples,:].reshape(-1,n_features+1)
            ind += n_cached_ind[j][0]
    
    return cached_dataset

def cache_imgs(A,obs_dataset,y_pred,n_cache,n_device,n_class,n_features):
    cached_dataset = torch.zeros((n_cache*n_device,n_features+1))
    ind = 0
    for i in range(n_device):
        n_cached_ind = A[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(y_pred[i]==j).tolist(), k= n_cached_ind[j])  
            cached_dataset[ind:ind+n_cached_ind[j],:] = obs_dataset[i][ind_samples,:].reshape(-1,n_features+1)
            ind += n_cached_ind[j]
    
    return cached_dataset

def cache_imgs_MNIST(A,obs_dataset,y_pred,n_cache,n_device,n_class):
    ind = 0
    Xs, ys = torch.zeros((n_cache*n_device,28,28),dtype=torch.float32), torch.zeros((n_cache*n_device),dtype=torch.int64)

    for i in range(n_device):
        n_cached_ind = A[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(y_pred[i]==j).reshape(-1).tolist(), k= n_cached_ind[j][0])
            for vv in ind_samples:
                Xs[ind,:],ys[ind] = obs_dataset[i][vv][0],obs_dataset[i][vv][1]
                ind += 1
    return create_MNIST_dataset(Xs,ys)

def cache_imgs_CIFAR(A,obs_dataset,y_pred,n_cache,n_device,n_class):
    ind = 0
    Xs, ys = torch.zeros((n_cache*n_device,32,32,3),dtype=torch.uint8), torch.zeros((n_cache*n_device),dtype=torch.int64)

    for i in range(n_device):
        n_cached_ind = A[i*n_class:(i+1)*n_class]
        for j in range(n_class):
            ind_samples = random.sample(np.argwhere(y_pred[i]==j).reshape(-1).tolist(), k= n_cached_ind[j][0])
            for vv in ind_samples:
                Xs[ind,:],ys[ind] = obs_dataset[i][vv][0].permute(1,2,0),obs_dataset[i][vv][1]
                ind += 1
    
    return create_CIFAR10_traindataset(Xs,ys)

def load_MNIST_dataset(save_dir,val_ratio):

    mnist_trainset = datasets.MNIST(root=save_dir, train=True, download=True)
    mnist_testset = datasets.MNIST(root=save_dir, train=False, download=True)

    X_train = mnist_trainset.data
    y_train = mnist_trainset.targets
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    X_test = mnist_testset.data
    y_test = mnist_testset.targets

    return X_train, X_test, X_val, y_train, y_test, y_val

def load_CIFAR10_dataset(save_dir,val_ratio):

    cifar_trainset = datasets.CIFAR10(root=save_dir,train=True,download=True)
    cifar_testset = datasets.CIFAR10(root=save_dir,train=False,download=True)

    X_train = torch.tensor(cifar_trainset.data)
    y_train = torch.tensor(cifar_trainset.targets,dtype=int)

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    X_test = torch.tensor(cifar_testset.data)
    y_test = torch.tensor(cifar_testset.targets,dtype=int)

    return X_train, X_test, X_val, y_train, y_test, y_val

def load_CIFAR100_dataset(save_dir,val_ratio):

    cifar_trainset = datasets.CIFAR100(root=save_dir,train=True,download=True)
    cifar_testset = datasets.CIFAR100(root=save_dir,train=False,download=True)

    X_train = torch.tensor(cifar_trainset.data)
    y_train = torch.tensor(cifar_trainset.targets,dtype=int)

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    X_test = torch.tensor(cifar_testset.data)
    y_test = torch.tensor(cifar_testset.targets,dtype=int)

    return X_train, X_test, X_val, y_train, y_test, y_val

def load_AdverseWeather_labels(weather_loc,daytime_loc,test_ratio,val_ratio,cache_all=False):
    
    with open(weather_loc,"r") as file:
        weather_labels = yaml.safe_load(file)
    
    with open(daytime_loc,"r") as file:
        daytime_labels = yaml.safe_load(file)
    
    assert list(daytime_labels.keys()).sort()==list(weather_labels.keys()).sort() ,"Labels don't match."

    img_locs = list(daytime_labels.keys())

    if cache_all:
        w = 455
        h = 256    
        X = torch.zeros((len(img_locs),256,455,3),dtype=torch.uint8)
        sim_bar = tqdm(img_locs,total=len(img_locs))
        for ind,i in enumerate(sim_bar):
            Img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
            X[ind] = torch.from_numpy(cv2.resize(Img,(w,h)))
    else:
        X = img_locs

    y = ["".join(weather_labels[x])+daytime_labels[x] for x in img_locs]

    classes = list(set(y))

    classes.sort()

    y = torch.tensor(list(map(lambda x:classes.index(x),y)),dtype=int)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_ratio,stratify=y,random_state=1)

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    return X_train, X_test, X_val, y_train, y_test, y_val, len(classes)

def load_BDD_labels(label_loc,img_loc,val_ratio,cache_all=False):
    
    with open(label_loc+"/det_train.json","r") as file:
        train_labels = json.load(file)
    
    with open(label_loc+"/det_val.json","r") as file:
        val_labels = json.load(file)

    label_map = {
    "rainy":0,
    "snowy":1,
    "foggy": 2,
    "overcast":3,
    "partly cloudy": 4,
    "clear": 5,
    "undefined":6
    }

    w_val_labels = list(map(lambda x:x["attributes"]["weather"],val_labels))
    w_train_labels = list(map(lambda x:x["attributes"]["weather"],train_labels)) 

    list(map(lambda x:img_loc+"/val/"+x["name"],val_labels))

    train_locs = list(map(lambda x:img_loc+"/train/"+x["name"],train_labels))
    test_locs = list(map(lambda x:img_loc+"/val/"+x["name"],val_labels))
    
    if cache_all:

        sim_bar = tqdm(train_locs,total=len(train_locs))
        X_train = torch.zeros((len(train_locs),256,455,3),dtype=torch.uint8)
        w = 455
        h = 256   
        for ind,i in enumerate(sim_bar):
            Img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
            X_train[ind] = torch.from_numpy(cv2.resize(Img,(w,h)))
        
        sim_bar = tqdm(test_locs,total=len(test_locs))
        X_test = torch.zeros((len(test_locs),256,455,3),dtype=torch.uint8)
        for ind,i in enumerate(sim_bar):
            Img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
            X_test[ind] = torch.from_numpy(cv2.resize(Img,(w,h)))
        
    else:
        X_train = train_locs
        X_test = test_locs

    y_train = torch.tensor(list(map(lambda x:label_map[x],w_train_labels)),dtype=int)
    y_test = torch.tensor(list(map(lambda x:label_map[x],w_val_labels)),dtype=int)

    filt_test = (y_test!=6).tolist()
    filt_train = (y_train!=6).tolist()

    y_train = y_train[filt_train]    
    y_test = y_test[filt_test]

    if cache_all:
        X_train = X_train[filt_train]
        X_test = X_test[filt_test]
    else:
        X_test = [X_test[i] for i in range(len(filt_test)) if filt_test[i]]
        X_train = [X_train[i] for i in range(len(filt_train)) if filt_train[i]]

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    return X_train, X_test, X_val, y_train, y_test, y_val, 6

def create_MNIST_dataset(X,y):
    transform = trfm.Compose([
    trfm.ToTensor(),
    trfm.Normalize((0.1307), (0.3081))])
    dataset = MNISTDataset(X,y,transform)
    return dataset

def create_CIFAR10_dataset(X,y):
    transform = trfm.Compose([
    trfm.ToTensor(),
    trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = CIFAR10Dataset(X,y,transform)
    return dataset

def create_AdverseWeather_dataset(X,y,cache_all=False):
    transform  = trfm.Compose([
        trfm.Resize(256),
        trfm.RandomCrop(224),
        trfm.ToTensor(),
        trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    dataset = AdverseWeatherDataset(X,y,transform,cache_all)
    return dataset

def create_CIFAR10_traindataset(X,y):
    transform = trfm.Compose([
    trfm.RandomCrop(32, padding=4),
    trfm.RandomHorizontalFlip(),
    trfm.ToTensor(),
    trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])
    dataset = CIFAR10Dataset(X,y,transform)
    return dataset

def plot_KLs(KL_coop,KL_ind,KL_feed,KL_unif,KL_lwb,save_loc):

    coop_df = pd.DataFrame(KL_coop.reshape(-1,1),columns=["L2"])
    coop_df["Round"] = [i for i in range(KL_coop.shape[1])]*KL_coop.shape[0]
    ind_df = pd.DataFrame(KL_ind.reshape(-1,1),columns=["L2"])
    ind_df["Round"] = [i for i in range(KL_ind.shape[1])]*KL_ind.shape[0]
    feed_df = pd.DataFrame(KL_feed.reshape(-1,1),columns=["L2"])
    feed_df["Round"] = [i for i in range(KL_feed.shape[1])]*KL_feed.shape[0]
    unif_df = pd.DataFrame(KL_unif.reshape(-1,1),columns=["L2"])
    unif_df["Round"] = [i for i in range(KL_unif.shape[1])]*KL_unif.shape[0]
    lwb_df = pd.DataFrame(KL_lwb.reshape(-1,1),columns=["L2"])
    lwb_df["Round"] = [i for i in range(KL_lwb.shape[1])]*KL_lwb.shape[0]

    fig, ax = plt.subplots(figsize=(7,7),dpi=600)

    sns.lineplot(data=coop_df,x="Round",y="L2",label="Oracle",linewidth=3,linestyle='-')
    sns.lineplot(data=ind_df,x="Round",y="L2",label="Greedy",linewidth=3,linestyle='-.')
    sns.lineplot(data=feed_df,x="Round",y="L2",label="Interactive",linewidth=3,linestyle='--')
    sns.lineplot(data=unif_df,x="Round",y="L2",label="Uniform",linewidth=3,linestyle='-')
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
    plt.savefig(save_loc)

def plot_Accs(coop_acc,ind_acc,feed_acc,unif_acc,save_loc,plt_int=False):

    acc_df = pd.DataFrame(np.concatenate((ind_acc[:,0],coop_acc[:,0],feed_acc[:,0],unif_acc[:,0],ind_acc[:,1],coop_acc[:,1],feed_acc[:,1],unif_acc[:,1])),columns=["Acc"])

    max_y_lim = min(max(acc_df["Acc"]) + .05,1)
    min_y_lim = max(min(acc_df["Acc"])-0.05,0)
    
    n_sim = ind_acc.shape[0]
    acc_df["Policy"] = ["Greedy"]*n_sim + ["Oracle"]*n_sim + ["Interactive"]*n_sim + ["Uniform"]*n_sim + ["Greedy"]*n_sim + ["Oracle"]*n_sim + ["Interactive"]*n_sim + ["Uniform"]*n_sim 
    acc_df["Type"] = ["Initial"]*4*n_sim + ["Final"]*4*n_sim
    
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(7,7),dpi=600)
    order = ['Greedy', 'Interactive', 'Oracle', 'Uniform']
    gg = sns.boxplot(data=acc_df,x="Policy",y="Acc",hue="Type",palette=["r","b"],order=order)
    add_stat_annotation(gg, data=acc_df, x="Policy", y="Acc",hue="Type", order=order,
                    box_pairs=[(("Greedy", "Final"), ("Interactive", "Final"))],
                     test='t-test_ind', text_format='full', loc='outside', verbose=2)
    # test='Mann-Whitney', text_format='star', verbose=2,loc="inside"
    # for legend text
    plt.setp(gg.get_legend().get_texts(), fontsize='14') 
 
    # for legend title
    plt.setp(gg.get_legend().get_title(), fontsize='18') 
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.ylim(min_y_lim, max_y_lim)
    plt.legend(loc='lower left',prop={'size': 14})
    plt.tight_layout()
    plt.savefig(save_loc)

def create_labels_Adverse_Weather(dataset_loc,out_loc,label_map,frame_per_image):
    
    daytime_label = dict()
    weather_label = dict()
    category_num = []
    image_locs = []

    for (dirpath,dirnames,filenames) in os.walk(dataset_loc):
        filenames.sort(key=lambda x: int(x.split(".")[0][3:]))
        i = 0
        for filename in filenames:
            if i%frame_per_image == 0:
                image_locs.append(os.path.join(dirpath,filename))
            i +=1

    for i in image_locs:
        dirs =  i.split("/")
        category_num.append(int(dirs[-4]))
        daytime_label[i] = label_map[int(dirs[-4])]["Daytime"]
        weather_label[i] = label_map[int(dirs[-4])]["Weather"]
    
    with open(out_loc+'/daytime_labels.yml', 'w') as outfile:
        yaml.dump(daytime_label, outfile, default_flow_style=False)
    with open(out_loc+'/weather_labels.yml', 'w') as outfile:
        yaml.dump(weather_label, outfile, default_flow_style=False)
    
def create_MNIST_dataloader(dataset, b_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0)
    data = torch.zeros((b_size,1, 28,28), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    return dataloader, data, label

def create_CIFAR_dataloader(dataset, b_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0)
    data = torch.zeros((b_size,3, 32,32), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    return dataloader, data, label

def train_MNIST_model(model, losses, loss_fn, optimizer, num_epoch, data, label, dataloader, silent=True):
    model.train()

    if not silent:
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)
    else:
        pbar = [i for i in range(num_epoch)]
    for epoch in pbar:
        for x,y in dataloader:
            data[:] = x
            label[:] = y

            model.zero_grad()

            out = model(data)
            loss = loss_fn(out, label)
            losses.append(loss)

            loss.backward()

            optimizer.step()
        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    return losses

def train_CIFAR_model(model, losses, loss_fn, optimizer, num_epoch, data, label, dataloader, silent=True):
    model.train()

    if not silent:
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)
    else:
        pbar = [i for i in range(num_epoch)]
    for epoch in pbar:
        for x,y in dataloader:
            data[:] = x.reshape(data.shape)
            label[:] = y    

            model.zero_grad()

            out = model(data)
            loss = loss_fn(out, label)
            losses.append(loss)

            loss.backward()

            optimizer.step()
        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    return losses

def test_MNIST_function(model, test_dataset, device):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    data = torch.zeros((len(test_dataset), 1, 28, 28), dtype=torch.float).to(device)
    label = torch.zeros((len(test_dataset)), dtype=int).to(device)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    label_stats = np.zeros(10,dtype=int)
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:

            data[:] = x.reshape(data.shape)
            label[:] = y

            out = model(data)

            _, pred = torch.max(out.data, 1)

            for i in range(10):
                filt_i = (label == i)
                pred_i = pred[filt_i]
                label_stats[i] += sum(filt_i) 
                for j in range(10):
                    filt_j = (pred_i == j)
                    nnum = sum(filt_j)
                    confusion_matrix[i, j] += nnum

    return confusion_matrix

def test_CIFAR_function(model, test_dataset, device,b_size):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size)
    data = torch.zeros((b_size, 3, 32, 32), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:

            data[:] = x.reshape(data.shape)
            label[:] = y

            out = model(data)

            _, pred = torch.max(out.data, 1)

            for i in range(10):
                filt_i = (label == i)
                pred_i = pred[filt_i]
                for j in range(10):
                    filt_j = (pred_i == j)
                    nnum = sum(filt_j)
                    confusion_matrix[i, j] += nnum

    return confusion_matrix

def estimate_xpred(N_y_hat,P_cond,n_device,n_class,x_hat_prev,alpha):

    size = np.sum(N_y_hat,axis=1)[0] 

    x_hat = np.zeros((n_device,n_class))
    x_dist = np.zeros((n_class),float)
    for i in range(n_device):
        x = cp.Variable((n_class,1))
        act_mat = np.eye(n_class)
        eq_mat = np.ones((1,n_class))
        b_eq = np.ones((1,1))
        b = np.zeros((n_class,1))
        constraint = [act_mat @ x >= b, eq_mat @ x == b_eq ]
        obj = cp.Minimize(cp.norm((N_y_hat[i]/size).reshape(-1,1) - P_cond[i] @ x ))
        prob = cp.Problem(obj, constraint)
        prob.solve()
        x_dist[:] = x.value.reshape(-1)
        target = np.array(saferound((x_dist*size).tolist(),places=0)).reshape(-1)/ size
        x_hat[i,:] = target
    
    x_hat = x_hat_prev + alpha *(x_hat - x_hat_prev)
    
    return x_hat

def create_run_dir(run_loc):

    run_i = 0
    if os.path.exists(run_loc):
        exps = os.listdir(run_loc)
    else:
        os.makedirs(run_loc)
        exps = []
    
    for i in range(len(exps)):
        if "run"+str(run_i) in exps:
            run_i += 1
        else:
            break

    os.makedirs(run_loc+"/run"+str(run_i))
    return run_loc+"/run"+str(run_i)

def ind_datasets(y_tr,y_tt,y_val):
    ind_tt = [i for i in range(len(y_tt))]
    ind_tr = [i for i in range(len(y_tr))]
    ind_val = [i for i in range(len(y_val))]
    return ind_tr,ind_tt,ind_val

def init_KLs(n_sim,n_rounds):
    KL_ind = np.zeros((n_sim,n_rounds+1))
    KL_coop = np.zeros((n_sim,n_rounds+1))
    KL_feed = np.zeros((n_sim,n_rounds+1))
    KL_unif = np.zeros((n_sim,n_rounds+1))
    KL_lwb = np.zeros((n_sim,n_rounds+1))

    return KL_ind,KL_coop,KL_feed,KL_unif,KL_lwb

def combine_sims(run_ids,run_loc,target_loc,CIFAR100=False):

    target_run_loc = create_run_dir(target_loc)

    with open(run_loc+"/run"+str(run_ids[0])+"/unif_params.yml") as f:
        params = yaml.safe_load(f)

    KL = dict()
    Acc = dict()
    Min_Acc = dict()
    Acc_Matrix = dict()
    top5_Acc = dict()

    sim_types = ["unif","lwb","coop","feed","ind"]

    for sim_type in sim_types:

        params = dict()
        with open(run_loc+"/run"+str(run_ids[0])+"/"+sim_type+"_params.yml") as f:
            params = yaml.safe_load(f)
        seeds = list()
        obs_ind = dict()
        dataset_ind = dict()
        KL[sim_type] = np.zeros((len(run_ids),params["n_rounds"]+1))
        Acc[sim_type] = np.zeros((len(run_ids),2))
        Min_Acc[sim_type] = np.zeros((len(run_ids),2))
        Acc_Matrix[sim_type] = np.zeros((len(run_ids),2,params["n_class"],params["n_class"]))
        top5_Acc[sim_type] = np.zeros((len(run_ids),2))
        tot_sim = 0
        
        for i,run_i in enumerate(run_ids):
            

            with open(run_loc+"/run"+str(run_i)+"/"+sim_type+"_params.yml") as f:
                new_params = yaml.safe_load(f)
            
            if new_params != params:
                print("Error in run"+str(run_i)+". Params don't match.")
                continue

            with open(run_loc+"/run"+str(run_i)+"/"+sim_type+"_seeds.yml") as f:
                new_seed = yaml.safe_load(f)
                if sim_type == "coop":
                    shutil.copyfile(run_loc+"/run"+str(run_i)+"/init_model"+str(new_seed[0])+".pt", 
                    target_run_loc+"/init_model"+str(new_seed[0])+".pt")
            
            if any(s in seeds for s in new_seed):
                print("Error in run"+str(run_i)+". The sim seed is already added.")
                continue
            else:
                seeds += new_seed
            tot_sim += 1
            with open(run_loc+"/run"+str(run_i)+"/"+sim_type+"_obs_ind.yml") as f:
                new_obs_ind = yaml.safe_load(f)
            obs_ind.update(new_obs_ind)

            with open(run_loc+"/run"+str(run_i)+"/"+sim_type+"_dataset_ind.yml") as f:
                new_dataset_ind = yaml.safe_load(f)
            dataset_ind.update(new_dataset_ind)

            KL[sim_type][i,:] = np.load(run_loc+"/run"+str(run_i)+"/"+sim_type+"_KL.npy")

            if sim_type != "lwb":
                Acc[sim_type][i,:] = np.load(run_loc+"/run"+str(run_i)+"/"+sim_type+"_acc.npy")
                Min_Acc[sim_type][i,:] = np.load(run_loc+"/run"+str(run_i)+"/"+sim_type+"_min_acc.npy")
                Acc_Matrix[sim_type][i,:] = np.load(run_loc+"/run"+str(run_i)+"/"+sim_type+"_acc_matrix.npy")

                if CIFAR100:
                    top5_Acc[sim_type][i,:] = np.load(run_loc+"/run"+str(run_i)+"/"+sim_type+"_top5_acc.npy")

                shutil.copyfile(run_loc+"/run"+str(run_i)+"/"+sim_type+"_last_model"+str(new_seed[0])+".pt", 
                target_run_loc+"/"+sim_type+"_last_model"+str(new_seed[0])+".pt")
        
        params["n_sim"] = tot_sim
        
        with open(target_run_loc+'/'+sim_type+'_params.yml', 'w') as outfile:
            yaml.dump(params, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_obs_ind.yml', 'w') as outfile:
            yaml.dump(obs_ind, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_dataset_ind.yml', 'w') as outfile:
            yaml.dump(dataset_ind, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_seeds.yml', 'w') as outfile:
            yaml.dump(seeds, outfile, default_flow_style=False)

        with open(target_run_loc+'/'+sim_type+'_KL.npy', 'wb') as outfile:
            np.save(outfile, KL[sim_type])

        with open(target_run_loc+'/'+sim_type+'_acc.npy', 'wb') as outfile:
            np.save(outfile, Acc[sim_type])

        with open(target_run_loc+'/'+sim_type+'_min_acc.npy', 'wb') as outfile:
            np.save(outfile, Min_Acc[sim_type])
        
        if CIFAR100:
            with open(target_run_loc+'/'+sim_type+'_top5_acc.npy', 'wb') as outfile:
                np.save(outfile, top5_Acc[sim_type])

        with open(target_run_loc+'/'+sim_type+'_acc_matrix.npy', 'wb') as outfile:
            np.save(outfile, Acc_Matrix[sim_type])

    plot_KLs(KL["coop"],KL["ind"],KL["feed"],KL["unif"],KL["lwb"],target_run_loc+"/L2_Norm.jpg")

    plot_Accs(Acc["coop"],Acc["ind"],Acc["feed"],Acc["unif"],target_run_loc+"/Accs.jpg")

    plot_Accs(Min_Acc["coop"],Min_Acc["ind"],Min_Acc["feed"],Min_Acc["unif"],target_run_loc+"/Min_Accs.jpg")

    if CIFAR100:
        plot_Accs(top5_Acc["coop"],top5_Acc["ind"],top5_Acc["feed"],top5_Acc["unif"],target_run_loc+"/top5_Accs.jpg")

# Function to initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)

class MNIST_Unif_Sim:
    def __init__(self,params,device):
        
        self.n_device = params["n_device"]
        self.n_sim = params["n_sim"]
        self.n_rounds = params["n_rounds"]
        self.n_epoch = params["n_epoch"]
        self.b_size = params["b_size"]
        self.n_iter = params["n_iter"]
        self.n_class = params["n_class"]
        self.test_b_size = params["test_b_size"]
        self.lr = params["lr"]
        self.n_size = params["n_size"]
        self.n_obs = params["n_obs"]
        self.n_cache = params["n_cache"]

        self.params = params
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

        self.KLs = np.zeros((self.n_sim,self.n_rounds+1))
        self.accs = np.zeros((self.n_sim,2))
        self.accs_matrix = np.zeros((self.n_sim,2,self.n_class,self.n_class)) 
        self.min_accs = np.zeros((self.n_sim,2)) 

        self.obs_ind = dict()
        self.dataset_ind = dict()

        self.seeds = list()
        self.base_seeds = list()
        
        self.y_preds = dict()
        self.N_y_pred = np.zeros((self.n_device,self.n_class),int)
        self.sim_i = 0

        self.model = MNISTClassifier().to(self.device)
        self.model.apply(init_weights)

    def create_base_inds(self,y,base_classes,sim_i,seed):
        self.base_seeds.append(seed)
        np.random.seed(seed)
        inds = list()
        x_dist = np.random.rand(len(base_classes))
        x_dist = x_dist / sum(x_dist)
        
        #N_x = list(map(int,saferound(x_dist*self.n_size,places=0)))
        
        N_x = [0 for i in range(len(base_classes))]

        N_x_lims = np.zeros((len(base_classes),1),int)
        for i in range(len(base_classes)):
            N_x_lims[i] = torch.sum( y == base_classes[i]).item()

        n_lim = self.n_size
        args = [i for i in range(len(base_classes))]
        fulls = []
        while n_lim !=0:
            n_cache_cand = saferound(x_dist[~np.isin(np.arange(len(x_dist)), fulls)]*n_lim/sum(x_dist[~np.isin(np.arange(len(x_dist)), fulls)]),places=0)
            f = [arg for arg in args if (arg not in fulls)]
            for g,arg in enumerate(f): 
                if N_x_lims[arg]<=n_cache_cand[g]+N_x[arg]:
                    prev_nx = N_x[arg]
                    N_x[arg] = int(N_x_lims[arg])
                    fulls.append(arg)
                    n_lim = n_lim - int(N_x_lims[arg]-prev_nx)
                else:
                    N_x[arg] += int(n_cache_cand[g])
                    n_lim = n_lim - int(n_cache_cand[g]) 
              
        for i,c_i in enumerate(base_classes):
            ind_s = random.sample(np.argwhere(y == c_i)[0].tolist(), k = N_x[i])
            inds.extend(ind_s)
        self.dataset_ind[sim_i]= [inds]

    def create_unif_base_inds(self,y,base_classes,sim_i,seed):
        self.base_seeds.append(seed)
        np.random.seed(seed)
        inds = list()
        x_dist = np.ones(len(base_classes))
        x_dist = x_dist / sum(x_dist)

        #N_x = list(map(int,saferound(x_dist*self.n_size,places=0)))
        N_x = [0 for i in range(len(base_classes))]
        
        N_x_lims = np.zeros((len(base_classes),1),int)
        for i in range(len(base_classes)):
            N_x_lims[i] = torch.sum( y == base_classes[i]).item()

        n_lim = self.n_size
        args = [i for i in range(len(base_classes))]
        fulls = []
        while n_lim !=0:
            n_cache_cand = saferound(x_dist[~np.isin(np.arange(len(x_dist)), fulls)]*n_lim/sum(x_dist[~np.isin(np.arange(len(x_dist)), fulls)]),places=0)
            f = [arg for arg in args if (arg not in fulls)]
            for g,arg in enumerate(f): 
                if N_x_lims[arg]<=n_cache_cand[g]+N_x[arg]:
                    prev_nx = N_x[arg]
                    N_x[arg] = int(N_x_lims[arg])
                    fulls.append(arg)
                    n_lim = n_lim - int(N_x_lims[arg]-prev_nx)
                else:
                    N_x[arg] += int(n_cache_cand[g])
                    n_lim = n_lim - int(n_cache_cand[g]) 

        for i,c_i in enumerate(base_classes):
            ind_s = random.sample(np.argwhere(y == c_i)[0].tolist(), k = N_x[i])
            inds.extend(ind_s)
        self.dataset_ind[sim_i]= [inds]       

    def set_base_inds(self,inds,sim_i):
        self.dataset_ind[sim_i]= [inds]

    def train_model(self,train_dataset,silent=True):

        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
        
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.b_size, shuffle=True, worker_init_fn=0)
    
        if not silent:
            pbar = tqdm([i for i in range(self.n_epoch)], total=self.n_epoch)
        else:
            pbar = [i for i in range(self.n_epoch)]
        for epoch in pbar:
            for x,y in dataloader:
                
                self.model.zero_grad()

                out = self.model(x.to(self.device))
                loss = self.loss_fn(out, y.to(self.device))

                loss.backward()

                optimizer.step()

            lr_sch.step()

            if not silent:
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
                pbar.set_description(s)

    def test_model(self,test_dataset,top5=False):
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_b_size)
        confusion_matrix = np.zeros((self.n_class, self.n_class), dtype=int)
        label_stats = np.zeros(self.n_class,dtype=int)
        top5correct = 0

        self.model.eval()
        with torch.no_grad():
            for x,y in test_loader:
                    
                out = self.model(x.to(self.device))
                _, pred = torch.max(out.data, 1)
                vals,inds = torch.topk(out.data, 5)
                top5correct += torch.sum(inds == y.to(self.device).repeat(5,1).T).item()
                for i in range(self.n_class):

                    filt_i = (y == i)
                    label_stats[i] += sum(filt_i)
                    pred_i = pred[filt_i]

                    for j in range(self.n_class):
                        filt_j = (pred_i == j)
                        nnum = sum(filt_j)
                        confusion_matrix[i, j] += nnum
        if top5:
            return confusion_matrix, label_stats, top5correct/len(test_dataset)
        else:
            return confusion_matrix, label_stats

    def cond_prob_calc(self,val_dataset):
        conf_matrix,label_stats = self.test_model(val_dataset)
        cond_prob = conf_matrix /(label_stats.reshape(1,-1)+1e-10)
        return cond_prob

    def acc_calc(self,conf_matrix,label_stats):
        return np.trace(conf_matrix)/np.sum(label_stats)

    def acc_matrix_calc(self,conf_matrix,label_stats):
        return (conf_matrix.T/label_stats).T

    def min_acc_calc(self,conf_matrix):
        return min(np.diag(conf_matrix))

    def save_model(self,save_loc,name):
        
        isExist = os.path.exists(save_loc)

        if not isExist:
            os.makedirs(save_loc)
        
        torch.save(self.model.state_dict(), save_loc+"/"+name)

    def load_model(self,loc):
        self.model.load_state_dict(torch.load(loc))

    def dataset_stats(self,y_train):
        N_x = np.zeros((self.n_class,1),int)

        labels = y_train[self.dataset_ind[self.sim_seed][-1]]
        for i in range(self.n_class):
            N_x[i] += sum(labels==i).item()
        return N_x

    def create_dataset(self,X,y):
        transform = trfm.Compose([
        trfm.ToTensor(),
        trfm.Normalize((0.1307), (0.3081))])
        dataset = MNISTDataset(X,y,transform)
        return dataset

    def create_traindataset(self,X,y):
        transform = trfm.Compose([
        trfm.ToTensor(),
        trfm.Normalize((0.1307), (0.3081))])
        dataset = MNISTDataset(X,y,transform)
        return dataset

    def y_pred_stats(self,y_pred):
        N_x = np.zeros((self.n_class,1),int)
        for i in range(self.n_class):
            N_x[i] = np.sum(y_pred == i)
        return N_x

    def eval_obs(self,dataset):
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        self.model.eval()
        y_pred = np.zeros(len(dataset),int)
        i =0
        with torch.no_grad():
            for x,_ in test_loader:

                out = self.model(x.to(self.device))

                _, pred = torch.max(out.data, 1)

                y_pred[:] = pred.cpu().numpy()
        return y_pred

    def create_xdist(self,sim_i,obs_clss,y):
        
        np.random.seed(sim_i)

        x_dist = np.zeros((self.n_device,self.n_class))
        for i in range(self.n_device):
            for j in obs_clss[i]:
                x_dist[i,j] = np.random.rand()
        
        x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

        N_x_lims = np.zeros((self.n_class,1),int)
        for i in range(self.n_class):
            N_x_lims[i] = torch.sum( y == i).item()

        N_x = np.zeros((self.n_device,self.n_class),dtype=int)
        for i in range(self.n_device):
            n_lim = self.n_obs
            args = [arg for arg in range(self.n_class)]
            fulls = []
            while n_lim!=0:
                n_cache_cand = saferound(x_dist[i,~np.isin(np.arange(len(x_dist[i])), fulls)]*n_lim/sum(x_dist[i,~np.isin(np.arange(len(x_dist[i])), fulls)]),places=0)
                f = [arg for arg in args if (arg not in fulls)]
                for g,arg in enumerate(f): 
                    if N_x_lims[arg]<=n_cache_cand[g]+N_x[i,arg]:
                        prev_nx = N_x[i,arg]
                        N_x[i,arg] = int(N_x_lims[arg])
                        fulls.append(arg)
                        n_lim = n_lim - int(N_x_lims[arg]-prev_nx)
                    else:
                        N_x[i,arg] += int(n_cache_cand[g])
                        n_lim = n_lim - int(n_cache_cand[g]) 
            x_dist[i] = N_x[i,:]
        
        x_dist = x_dist / np.sum(x_dist,1).reshape(-1,1)

        return x_dist, N_x

    def create_obs_ind(self,N_x,y,sim_i):
    
        np.random.seed(sim_i)
        random.seed(sim_i)

        obs_ind = [[] for i in range(self.n_rounds)]

        for round_i in range(self.n_rounds):
            obs_ind[round_i] = [[] for i in range(self.n_device)]
            for dev_i in range(self.n_device):
                for i in range(self.n_class):
                    ind_s = random.sample(np.argwhere(y == i).tolist()[0], k = N_x[dev_i][i])
                    obs_ind[round_i][dev_i].extend(ind_s)
        
        return obs_ind

    def cache_inds(self,A,round_i):
        inds = list()
        for i in range(self.n_device):
            n_cached_ind = A[i*self.n_class:(i+1)*self.n_class]
            for j in range(self.n_class):
                ind_samples = random.sample(np.argwhere(self.y_preds[i]==j).reshape(-1).tolist(), k= n_cached_ind[j][0])
                for ind in ind_samples:
                    inds.append(self.obs_ind[self.sim_seed][round_i][i][ind])
        return inds

    def Action(self,D_0_stats,D_target,P_cond,x_dist):

        A = np.zeros((self.n_device*self.n_class,1),int)

        for i in range(self.n_device):
            n_lim = self.n_cache
            args = [i for i in range(self.n_class)]
            random.shuffle(args)
            fulls = []
            while n_lim !=0:
                rem_len = self.n_class - len(fulls)
                n_cache_cand = saferound(np.ones(rem_len)*n_lim/rem_len,places=0)
                f = [arg for arg in args if (arg not in fulls)]
                for g,arg in enumerate(f): 
                    if self.N_y_pred[i][arg]<=n_cache_cand[g]+A[i*self.n_class+arg]:
                        prev_a = A[i*self.n_class+arg][0]
                        A[i*self.n_class+arg] = int(self.N_y_pred[i][arg])
                        fulls.append(arg)
                        n_lim = n_lim - int(self.N_y_pred[i][arg] - prev_a)
                    else:
                        A[i*self.n_class+arg] += int(n_cache_cand[g])
                        n_lim = n_lim - int(n_cache_cand[g])
        return A

    def sim_round(self,sim_i,sim_seed,X_train,y_train,val_dataset, base_inds,obs_ind,x_dist):
        self.sim_i = sim_i
        self.sim_seed = sim_seed
        self.obs_ind[sim_seed] = obs_ind

        random.seed(sim_seed)
        np.random.seed(sim_seed)
        self.seeds.append(sim_seed)

        D_target = np.ones((self.n_class,1))*self.n_size/self.n_class

        self.set_base_inds(base_inds[0],sim_seed)

        D_0_stats = self.dataset_stats(y_train)

        self.KLs[sim_i,0] = cp.norm(D_0_stats-D_target).value/sum(D_target)

        P_cond = self.cond_prob_calc(val_dataset)

        for round_i in range(self.n_rounds):

            D_target = np.ones((self.n_class,1))*(sum(D_target) + self.n_cache*self.n_device)/self.n_class
        
            for i in range(self.n_device):
                obs_dataset = self.create_dataset(X_train[self.obs_ind[sim_seed][round_i][i]],y_train[self.obs_ind[sim_seed][round_i][i]])
                self.y_preds[i] = self.eval_obs(obs_dataset)
                self.N_y_pred[i,:] = self.y_pred_stats(self.y_preds[i]).reshape(-1)
            
            A = self.Action(D_0_stats,D_target,P_cond,x_dist)
            
            cached_ind = self.cache_inds(A,round_i)

            self.dataset_ind[sim_seed].append(self.dataset_ind[sim_seed][-1] + cached_ind)

            D_0_stats = self.dataset_stats(y_train)

            self.KLs[sim_i,round_i+1] = cp.norm(D_0_stats-D_target).value/sum(D_target)
        
    def sim_acc(self,X_train,y_train,test_dataset):
        
        test_matrix,labels_stats = self.test_model(test_dataset)
        self.accs[self.sim_i,0] = self.acc_calc(test_matrix,labels_stats)
        self.accs_matrix[self.sim_i,0] = self.acc_matrix_calc(test_matrix,labels_stats)
        self.min_accs[self.sim_i,0] = self.min_acc_calc(self.accs_matrix[self.sim_i,0])
        
        self.model.apply(init_weights)
        train_dataset = self.create_traindataset(X_train[self.dataset_ind[self.sim_seed][-1]],y_train[self.dataset_ind[self.sim_seed][-1]])
        self.train_model(train_dataset)

        test_matrix,labels_stats = self.test_model(test_dataset)
        self.accs[self.sim_i,1] = self.acc_calc(test_matrix,labels_stats)
        self.accs_matrix[self.sim_i,1] = self.acc_matrix_calc(test_matrix,labels_stats)
        self.min_accs[self.sim_i,1] = self.min_acc_calc(self.accs_matrix[self.sim_i,1])

    def save_infos(self,save_loc,sim_type):

        with open(save_loc+'/'+sim_type+'_params.yml', 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)
    
        with open(save_loc+'/'+sim_type+'_dataset_ind.yml', 'w') as outfile:
            yaml.dump(self.dataset_ind, outfile, default_flow_style=False)

        with open(save_loc+'/'+sim_type+'_obs_ind.yml', 'w') as outfile:
            yaml.dump(self.obs_ind, outfile, default_flow_style=False)

        with open(save_loc+'/'+sim_type+'_seeds.yml', 'w') as outfile:
            yaml.dump(self.seeds, outfile, default_flow_style=False)

        with open(save_loc+'/'+sim_type+'_KL.npy', 'wb') as outfile:
            np.save(outfile, self.KLs)

        with open(save_loc+'/'+sim_type+'_acc.npy', 'wb') as outfile:
            np.save(outfile, self.accs)

        with open(save_loc+'/'+sim_type+'_acc_matrix.npy', 'wb') as outfile:
            np.save(outfile, self.accs_matrix)

        with open(save_loc+'/'+sim_type+'_min_acc.npy', 'wb') as outfile:
            np.save(outfile, self.min_accs)

class MNIST_Coop_Sim(MNIST_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    def Action(self,D_0_stats,D_target,P_cond,x_dist):

        A = np.zeros((self.n_device*self.n_class,1),int)

        P_cond = P_cond.reshape(1,self.n_class,self.n_class)

        P_cond = np.repeat(P_cond,self.n_device,axis=0)

        P_occ = x_dist.reshape(self.n_device,-1,1) * P_cond

        P_condr = P_occ /(np.sum(P_occ,1).reshape(self.n_device,1,-1)+1e-10)

        P_tuple = [P_condr[i] for i in range(self.n_device)]
        P_Condr = np.concatenate(P_tuple,axis=1)

        Act = cp.Variable((self.n_class*self.n_device,1))
        
        Act_mat = np.concatenate((np.eye(self.n_class*self.n_device),-np.eye(self.n_class*self.n_device)),axis=0)
        eq_mat = np.repeat(np.eye(self.n_device),self.n_class,axis=1)
        b_eq = np.ones((self.n_device,1))*self.n_cache
        B = np.concatenate((np.zeros((self.n_class*self.n_device,1)),-self.N_y_pred.reshape(-1,1)),axis=0)
        constraint = [Act_mat @ Act >= B, eq_mat @ Act == b_eq]
        obj = cp.Minimize(cp.sum_squares(D_0_stats + P_Condr @ Act - D_target))
        prob = cp.Problem(obj, constraint)

        prob.solve( )

        for i in range(self.n_device):
            act = Act.value[i*self.n_class:(i+1)*self.n_class]
            A[i*self.n_class:(i+1)*self.n_class] = np.array(saferound(act.reshape(-1),places=0),int).reshape(-1,1)
    
        return A

class MNIST_Ind_Sim(MNIST_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
    
    def Action(self,D_0_stats,D_target,P_cond,x_dist):

        A = np.zeros((self.n_device*self.n_class,1),int)

        P_cond = P_cond.reshape(1,self.n_class,self.n_class)

        P_cond = np.repeat(P_cond,self.n_device,axis=0)

        P_occ = x_dist.reshape(self.n_device,-1,1) * P_cond

        P_condr = P_occ /(np.sum(P_occ,1).reshape(self.n_device,1,-1)+1e-10)

        for i in range(self.n_device):
            act = cp.Variable((self.n_class,1))
            act_mat = np.concatenate((np.eye(self.n_class),-np.eye(self.n_class)),axis=0)
            eq_mat = np.ones((1,self.n_class))
            b_eq = np.ones((1,1))*self.n_cache
            b = np.concatenate((np.zeros((self.n_class,1)),-self.N_y_pred[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]
            
            obj = cp.Minimize(cp.sum_squares(D_0_stats+ P_condr[i] @ act -D_target))
            prob = cp.Problem(obj, constraint)
            prob.solve( )
            A[i*self.n_class:(i+1)*self.n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

        return A

class MNIST_Feed_Sim(MNIST_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
    
    def Action_Ind(self,D_0_stats,D_target,P_cond,x_dist):

        A = np.zeros((self.n_device*self.n_class,1),int)

        P_cond = P_cond.reshape(1,self.n_class,self.n_class)

        P_cond = np.repeat(P_cond,self.n_device,axis=0)

        P_occ = x_dist.reshape(self.n_device,-1,1) * P_cond

        P_condr = P_occ /(np.sum(P_occ,1).reshape(self.n_device,1,-1)+1e-10)

        for i in range(self.n_device):
            act = cp.Variable((self.n_class,1))
            act_mat = np.concatenate((np.eye(self.n_class),-np.eye(self.n_class)),axis=0)
            eq_mat = np.ones((1,self.n_class))
            b_eq = np.ones((1,1))*self.n_cache
            b = np.concatenate((np.zeros((self.n_class,1)),-self.N_y_pred[i,:].reshape(-1,1)),axis=0)
            constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]
            
            obj = cp.Minimize(cp.sum_squares(D_0_stats+ P_condr[i] @ act -D_target))
            prob = cp.Problem(obj, constraint)
            prob.solve( )
            A[i*self.n_class:(i+1)*self.n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

        return A

    def Action(self,D_0_stats,D_target,P_cond,x_dist):

        A_feed = self.Action_Ind(D_0_stats,D_target,P_cond,x_dist)

        P_cond = P_cond.reshape(1,self.n_class,self.n_class)

        P_cond = np.repeat(P_cond,self.n_device,axis=0)

        P_occ = x_dist.reshape(self.n_device,-1,1) * P_cond

        P_condr = P_occ /(np.sum(P_occ,1).reshape(self.n_device,1,-1)+1e-10)

        for _ in range(self.n_iter):
            for i in range(self.n_device):
                act = cp.Variable((self.n_class,1))
                act_mat = np.concatenate((np.eye(self.n_class),-np.eye(self.n_class)),axis=0)
                eq_mat = np.ones((1,self.n_class))
                b_eq = np.ones((1,1))*self.n_cache
                b = np.concatenate((np.zeros((self.n_class,1)),-self.N_y_pred[i,:].reshape(-1,1)),axis=0)
                constraint = [act_mat @ act >= b, eq_mat @ act == b_eq ]

                k = [l for l in range(i*self.n_class,(i+1)*self.n_class)]
                A_feed_o = A_feed[[j for j in range(self.n_device*self.n_class) if j not in k]]

                P_tuple = [P_condr[l] for l in range(self.n_device) if l!=i]
                P_Condr_o = np.concatenate(P_tuple,axis=1)

                obj = cp.Minimize(cp.sum_squares(D_0_stats+ np.matmul(P_Condr_o,A_feed_o) + P_condr[i] @ act- D_target))
        
                prob = cp.Problem(obj,constraint)
                prob.solve( )

                A_feed[i*self.n_class:(i+1)*self.n_class] = np.array(saferound(act.value.reshape(-1),places=0),int).reshape(-1,1) 

        return A_feed

class MNIST_Lwb_Sim(MNIST_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    def Action(self,D_0_stats,D_target):

        A_coop = np.zeros((self.n_device*self.n_class,1))
        A = np.eye(self.n_class).reshape(1,self.n_class,self.n_class).repeat(self.n_device,axis=0)
        A_tuple = [A[i] for i in range(self.n_device)]
        A = np.concatenate(A_tuple,axis=1)

        Act = cp.Variable((self.n_class*self.n_device,1))
        
        Act_mat = np.concatenate((np.eye(self.n_class*self.n_device),np.repeat(np.eye(self.n_device),self.n_class,axis=1),-np.repeat(np.eye(self.n_device),self.n_class,axis=1)),axis=0)
        
        B = np.concatenate((np.zeros((self.n_class*self.n_device,1)),np.ones((self.n_device,1))*self.n_cache,-np.ones((self.n_device,1))*self.n_cache),axis=0)
        constraint = [Act_mat @ Act - B >= 0]
        
        obj = cp.Minimize(cp.sum_squares(D_0_stats + A @ Act -D_target))
        prob = cp.Problem(obj, constraint)

        prob.solve( )
        A_coop[:] = Act.value
        A_m = np.array(saferound(np.matmul(A,A_coop).reshape(-1),places=0),int).reshape(-1,1)
        D_0_stats = D_0_stats + A_m
        
        return D_0_stats

    def sim_round(self,sim_i,sim_seed,X_train,y_train,val_dataset, base_inds,obs_ind,x_dist):
        self.sim_i = sim_i
        self.sim_seed = sim_seed
        self.obs_ind[sim_i] = obs_ind

        random.seed(sim_seed)
        np.random.seed(sim_seed)
        self.seeds.append(sim_seed)

        D_target = np.ones((self.n_class,1))*self.n_size/self.n_class

        self.set_base_inds(base_inds[0],sim_seed)

        D_0_stats = self.dataset_stats(y_train)

        self.KLs[sim_i,0] = cp.norm(D_0_stats-D_target).value/sum(D_target)

        for round_i in range(self.n_rounds):

            D_target = np.ones((self.n_class,1))*(sum(D_target) + self.n_cache*self.n_device)/self.n_class
        
            D_0_stats = self.Action(D_0_stats,D_target)

            self.KLs[sim_i,round_i+1] = cp.norm(D_0_stats-D_target).value/sum(D_target)

class CIFAR10_Unif_Sim(MNIST_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
        self.model =torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=False,verbose=False).to(device)
        self.model.apply(init_weights) 
    def create_dataset(self,X,y):
        transform = trfm.Compose([
        trfm.ToTensor(),
        trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset = CIFAR10Dataset(X,y,transform)
        return dataset

    def create_traindataset(self,X,y):
        transform = trfm.Compose([
        trfm.RandomCrop(32, padding=4),
        trfm.RandomHorizontalFlip(),
        trfm.ToTensor(),
        trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset = CIFAR10Dataset(X,y,transform)
        return dataset

class CIFAR10_Coop_Sim(CIFAR10_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Coop_Sim.Action

class CIFAR10_Ind_Sim(CIFAR10_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Ind_Sim.Action

class CIFAR10_Feed_Sim(CIFAR10_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Feed_Sim.Action
    Action_Ind = MNIST_Feed_Sim.Action_Ind

class CIFAR10_Lwb_Sim(CIFAR10_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Lwb_Sim.Action
    sim_round = MNIST_Lwb_Sim.sim_round

class CIFAR100_Unif_Sim(CIFAR10_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
        self.model =torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=False,verbose=False).to(device)  
        self.model.apply(init_weights)
        self.top5accs = np.zeros((self.n_sim,2))

    def sim_acc(self,X_train,y_train,test_dataset):
        
        test_matrix,labels_stats, self.top5accs[self.sim_i,0] = self.test_model(test_dataset,True)
        self.accs[self.sim_i,0] = self.acc_calc(test_matrix,labels_stats)
        self.accs_matrix[self.sim_i,0] = self.acc_matrix_calc(test_matrix,labels_stats)
        self.min_accs[self.sim_i,0] = self.min_acc_calc(self.accs_matrix[self.sim_i,0])
        
        self.model.apply(init_weights) 
        train_dataset = self.create_traindataset(X_train[self.dataset_ind[self.sim_seed][-1]],y_train[self.dataset_ind[self.sim_seed][-1]])
        self.train_model(train_dataset)

        test_matrix,labels_stats,self.top5accs[self.sim_i,1] = self.test_model(test_dataset,True)
        self.accs[self.sim_i,1] = self.acc_calc(test_matrix,labels_stats)
        self.accs_matrix[self.sim_i,1] = self.acc_matrix_calc(test_matrix,labels_stats)
        self.min_accs[self.sim_i,1] = self.min_acc_calc(self.accs_matrix[self.sim_i,1])
    
    def save_infos(self,save_loc,sim_type):

        with open(save_loc+'/'+sim_type+'_params.yml', 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)
    
        with open(save_loc+'/'+sim_type+'_dataset_ind.yml', 'w') as outfile:
            yaml.dump(self.dataset_ind, outfile, default_flow_style=False)

        with open(save_loc+'/'+sim_type+'_obs_ind.yml', 'w') as outfile:
            yaml.dump(self.obs_ind, outfile, default_flow_style=False)

        with open(save_loc+'/'+sim_type+'_seeds.yml', 'w') as outfile:
            yaml.dump(self.seeds, outfile, default_flow_style=False)

        with open(save_loc+'/'+sim_type+'_KL.npy', 'wb') as outfile:
            np.save(outfile, self.KLs)

        with open(save_loc+'/'+sim_type+'_acc.npy', 'wb') as outfile:
            np.save(outfile, self.accs)

        with open(save_loc+'/'+sim_type+'_top5_acc.npy', 'wb') as outfile:
            np.save(outfile, self.top5accs)

        with open(save_loc+'/'+sim_type+'_acc_matrix.npy', 'wb') as outfile:
            np.save(outfile, self.accs_matrix)

        with open(save_loc+'/'+sim_type+'_min_acc.npy', 'wb') as outfile:
            np.save(outfile, self.min_accs)

class CIFAR100_Coop_Sim(CIFAR100_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
        self.model =torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=False,verbose=False).to(device)  

    Action = MNIST_Coop_Sim.Action

class CIFAR100_Feed_Sim(CIFAR100_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
        self.model =torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=False,verbose=False).to(device)  

    Action = MNIST_Feed_Sim.Action
    Action_Ind = MNIST_Feed_Sim.Action_Ind

class CIFAR100_Ind_Sim(CIFAR100_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
        self.model =torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=False,verbose=False).to(device)  

    Action = MNIST_Ind_Sim.Action

class CIFAR100_Lwb_Sim(CIFAR100_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)
        self.model =torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=False,verbose=False).to(device)  

    Action = MNIST_Lwb_Sim.Action
    sim_round = MNIST_Lwb_Sim.sim_round

class AdverseWeather_Unif_Sim(MNIST_Unif_Sim):

    def __init__(self,params,device):
        super().__init__(params,device)
        self.model = vsmodels.resnet18(pretrained=True).to(device)
        self.model.fc = nn.Linear(in_features=512, out_features=7).to(device)

    def create_dataset(self,X,y,cache_all=False):
        transform  = trfm.Compose([
        trfm.Resize(256),
        trfm.CenterCrop(224),
        trfm.ToTensor(),
        trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        dataset = AdverseWeatherDataset(X,y,transform,cache_all)
        return dataset
        
    def create_traindataset(self,X,y,cache_all=True):
        transform = trfm.Compose([
        trfm.Resize(256),
        trfm.RandomCrop(224),
        trfm.RandomHorizontalFlip(),
        trfm.ToTensor(),
        trfm.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        dataset = AdverseWeatherDataset(X,y,transform,cache_all)
        return dataset
    
    def sim_round(self,sim_i,sim_seed,X_train,y_train,val_dataset, base_inds,obs_ind,x_dist,cache_all=False):
        self.sim_i = sim_i
        self.sim_seed = sim_seed
        self.obs_ind[sim_seed] = obs_ind

        random.seed(sim_seed)
        np.random.seed(sim_seed)
        self.seeds.append(sim_seed)

        D_target = np.ones((self.n_class,1))*self.n_size/self.n_class

        self.set_base_inds(base_inds[0],sim_seed)

        D_0_stats = self.dataset_stats(y_train)

        self.KLs[sim_i,0] = cp.norm(D_0_stats-D_target).value/sum(D_target)

        P_cond = self.cond_prob_calc(val_dataset)

        for round_i in range(self.n_rounds):

            D_target = np.ones((self.n_class,1))*(sum(D_target) + self.n_cache*self.n_device)/self.n_class
        
            for i in range(self.n_device):
                if cache_all:
                    obs_dataset = self.create_dataset(X_train[self.obs_ind[sim_seed][round_i][i]],y_train[self.obs_ind[sim_seed][round_i][i]],cache_all)
                else:    
                    obs_dataset = self.create_dataset([X_train[i] for i in self.obs_ind[sim_seed][round_i][i]],y_train[self.obs_ind[sim_seed][round_i][i]],cache_all)
                self.y_preds[i] = self.eval_obs(obs_dataset)
                self.N_y_pred[i,:] = self.y_pred_stats(self.y_preds[i]).reshape(-1)
            
            A = self.Action(D_0_stats,D_target,P_cond,x_dist)
            
            cached_ind = self.cache_inds(A,round_i)

            self.dataset_ind[sim_seed].append(self.dataset_ind[sim_seed][-1] + cached_ind)

            D_0_stats = self.dataset_stats(y_train)

            self.KLs[sim_i,round_i+1] = cp.norm(D_0_stats-D_target).value/sum(D_target)

    def sim_acc(self,X_train,y_train,test_dataset,cache_all=False):
        
        test_matrix,labels_stats = self.test_model(test_dataset)
        self.accs[self.sim_i,0] = self.acc_calc(test_matrix,labels_stats)
        self.accs_matrix[self.sim_i,0] = self.acc_matrix_calc(test_matrix,labels_stats)
        self.min_accs[self.sim_i,0] = self.min_acc_calc(self.accs_matrix[self.sim_i,0])

        self.model = vsmodels.resnet18(pretrained=True).to(self.device)
        if cache_all:
            train_dataset = self.create_traindataset(X_train[self.dataset_ind[self.sim_seed][-1]],y_train[self.dataset_ind[self.sim_seed][-1]],cache_all)
        else:
            train_dataset = self.create_traindataset([X_train[i] for i in self.dataset_ind[self.sim_seed][-1]],y_train[self.dataset_ind[self.sim_seed][-1]],cache_all)
        self.train_model(train_dataset)

        test_matrix,labels_stats = self.test_model(test_dataset)
        self.accs[self.sim_i,1] = self.acc_calc(test_matrix,labels_stats)
        self.accs_matrix[self.sim_i,1] = self.acc_matrix_calc(test_matrix,labels_stats)
        self.min_accs[self.sim_i,1] = self.min_acc_calc(self.accs_matrix[self.sim_i,1])

class AdverseWeather_Coop_Sim(AdverseWeather_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Coop_Sim.Action

class AdverseWeather_Feed_Sim(AdverseWeather_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Feed_Sim.Action
    Action_Ind = MNIST_Feed_Sim.Action_Ind

class AdverseWeather_Ind_Sim(AdverseWeather_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Ind_Sim.Action

class AdverseWeather_Lwb_Sim(AdverseWeather_Unif_Sim):
    def __init__(self,params,device):
        super().__init__(params,device)

    Action = MNIST_Lwb_Sim.Action
    sim_round = MNIST_Lwb_Sim.sim_round

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, device, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = optim.SGD(net.parameters(), lr=0.01,weight_decay=0.01)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=200, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')

    args = parser.parse_args()
    return args