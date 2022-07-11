import json


#labels_loc = "/store/datasets/bdd100k/labels"
labels_loc = "/store/datasets/bdd100k/labels/det_20"
imgs_loc = ""

#with open(labels_loc+"/bdd100k_labels_images_train.json") as file:
with open(labels_loc+"/det_train.json") as file:
    train_labels = json.load(file)

with open(labels_loc+"/det_val.json") as file:
    val_labels = json.load(file)

w_labels = list(map(lambda x:x["attributes"]["weather"],train_labels))

label_map = {
    "rainy":0,
    "snowy":1,
    "foggy": 2,
    "overcast":3,
    "partly cloudy": 4,
    "clear": 5,
    "undefined":6
}


s_labels = list(map(lambda x:x["attributes"]["scene"],train_labels))
t_labels = list(map(lambda x:x["attributes"]["timeofday"],train_labels))

ws_labels = list(map(lambda x,y:"".join([x,y]),w_labels,s_labels))
st_labels = list(map(lambda x,y:"".join([x,y]),s_labels,t_labels))
wt_labels = list(map(lambda x,y:"".join([x,y]),w_labels,t_labels))
c_labels = list(map(lambda x,y,z:"".join([x,y,z]),w_labels,s_labels,t_labels))

w_s = set(w_labels)
ws_s = set(ws_labels)
st_s = set(st_labels)
wt_s = set(wt_labels)
s_s = set(s_labels)
t_s = set(t_labels)
c_s = set(c_labels)


print("\nTotal number of weather class: "+str(len(w_s))+"\n")

for x in w_s:
    print('{} has occurred {} times'.format(x, w_labels.count(x)))

print("\nTotal number of scene class: "+str(len(s_s))+"\n")

for x in s_s:
    print('{} has occurred {} times'.format(x, s_labels.count(x)))

print("\nTotal number of timeofday class: "+str(len(t_s))+"\n")

for x in t_s:
    print('{} has occurred {} times'.format(x, t_labels.count(x)))

print("\nTotal number of weatherscene class: "+str(len(ws_s))+"\n")

for x in ws_s:
    print('{} has occurred {} times'.format(x, ws_labels.count(x)))

print("\nTotal number of scenetimeofday class: "+str(len(st_s))+"\n")

for x in st_s:
    print('{} has occurred {} times'.format(x, st_labels.count(x)))

print("\nTotal number of weathertimeofday class: "+str(len(wt_s))+"\n")

for x in wt_s:
    print('{} has occurred {} times'.format(x, wt_labels.count(x)))

print("\nTotal number of combined class: "+str(len(c_s))+"\n")

for x in c_s:
    print('{} has occurred {} times'.format(x, c_labels.count(x)))

