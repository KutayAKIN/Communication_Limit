import os,yaml
from sklearn.model_selection import train_test_split
import cv2
from utils import create_labels_Adverse_Weather

data_loc = "/home/oa5983/conv_opt/conv_opt/AdverseWeather/Recordings"
with open("/home/oa5983/conv_opt/conv_opt/AdverseWeather/label_maps.yml","r") as f:
    label_map = yaml.safe_load(f) 

frame_per_image = 5

create_labels_Adverse_Weather(data_loc,"/home/oa5983/conv_opt/conv_opt/AdverseWeather",label_map,frame_per_image)