import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *

X_train,X_test,X_val,y_train,y_test,y_val,n_class = load_AdverseWeather_labels("./AdverseWeather/weather_labels.yml","./AdverseWeather/daytime_labels.yml",0.1,0.1)

val_dataset = create_AdverseWeather_dataset(X_val,y_val)

test_dataset = create_AdverseWeather_dataset(X_test,y_test)

params = dict()

params["n_device"] = 20
params["n_sim"] = 1
params["n_rounds"] = 5
params["n_epoch"] = 100
params["b_size"] = 1024
params["n_iter"] = 1
params["n_class"] = n_class
params["test_b_size"] = 1000
params["lr"] = 0.1
params["n_size"] = 2048
params["n_obs"] = 2000
params["n_cache"] = 80

device = torch.device("cuda:5" if (torch.cuda.is_available()) else "cpu")

run_loc = "./runs/AdverseWeather/nums"

sim_i = 0
random.seed(sim_i)
torch.manual_seed(sim_i)
np.random.seed(sim_i)

base_classes =  [i for i in range(params["n_class"])]

Unif_Model = AdverseWeather_Unif_Sim(params,device)

Unif_Model.create_unif_base_inds(y_train,base_classes,sim_i,sim_i)

initial_dataset = Unif_Model.create_traindataset([X_train[i] for i in Unif_Model.dataset_ind[sim_i][0]],y_train[Unif_Model.dataset_ind[sim_i]])

Unif_Model.train_model(initial_dataset,False)

test_matrix,labels_stats = Unif_Model.test_model(test_dataset)
accs = Unif_Model.acc_calc(test_matrix,labels_stats)

print("Test Accuracy:",'{0:.3g}'.format(accs))
print(test_matrix)