from lib import *
from model import *
from eval import *
import os
import math


par_path = "/Users/admin/Documents/PFN/ml/param.txt"
#par_path = "/Users/tanakadaiki/Documents/PFN/ml/param.txt"

W1,W2,W3,b1,b2,b3 = get_model_param(par_path)
model_param = [W1,W2,W3,b1,b2,b3]

ad_path = "/Users/admin/Documents/PFN/ml/pgm_ad"

#advarsarial imagesにさらにadvarsarial images対する予測性能を測る
ad_ad_path = "/Users/admin/Documents/PFN/ml/pgm_ad_ad"

labels_path = "/Users/admin/Documents/PFN/ml/labels.txt"
true = get_true_label(labels_path)

epzero = 0.01

""" making advarasarial *2 images """
adv(ad_path, ad_ad_path, epzero, model_param, true)


acc = accuracy(ad_ad_path, true, model_param)
print(acc)
