from lib import *
from model import *
import os
import math
from eval import *


path = "/Users/admin/Documents/PFN/ml/pgm"
#path = "/Users/tanakadaiki/Documents/PFN/ml/pgm"
path_to_images = []
which_img = []

def sort(value):
    return int(value.split(".")[0])

sorted_path = sorted(os.listdir(path), key=sort)

for p in sorted_path:
    path_to_images.append(path + "/" + p)

""" getting parameters of model """
par_path = "/Users/admin/Documents/PFN/ml/param.txt"
#par_path = "/Users/tanakadaiki/Documents/PFN/ml/param.txt"
W1,W2,W3,b1,b2,b3 = get_model_param(par_path)
model_param = [W1,W2,W3,b1,b2,b3]

""" getting true labels """
labels_path = "/Users/admin/Documents/PFN/ml/labels.txt"
#labels_path = "/Users/tanakadaiki/Documents/PFN/ml/labels.txt"
true = get_true_label(labels_path)

acc = accuracy(path, true, model_param)

print("accuracy : " + str(acc))
