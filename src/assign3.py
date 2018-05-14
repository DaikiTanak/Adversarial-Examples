from lib import *
from model import Model
from eval import *

import os
import math


path = "/Users/admin/Documents/PFN/ml/pgm"
#path = "/Users/tanakadaiki/Documents/PFN/ml/pgm"
path_to_images = []


def sort(value):
    return int(value.split(".")[0])

sorted_path = sorted(os.listdir(path), key=sort)
for p in sorted_path:
    path_to_images.append(path + "/" + p)
features = list(map(tovector, path_to_images))

par_path = "/Users/admin/Documents/PFN/ml/param.txt"
W1,W2,W3,b1,b2,b3 = get_model_param(par_path)
model_param = [W1,W2,W3,b1,b2,b3]

labels_path = "/Users/admin/Documents/PFN/ml/labels.txt"
true = get_true_label(labels_path)

""" baseline """
ad_baseline = "/Users/admin/Documents/PFN/ml/pgm_ad_bl"
i = 1
epzero = 0.01
for f,t in zip(features, true):
    # images made by baseline method.
    model = Model(W1, b1, W2, b2, W3, b3, f)
    adv_bl_x = model.make_advarsarial_bl(epzero)
    adv_bl_image = list(map(lambda x:0 if(x<0) else min([255, int(x*255)]), adv_bl_x))
    filename = ad_baseline + "/" + str(i) + ".pgm"
    f = open(filename, "w")
    f.write(write_image(adv_bl_image))
    f.close()
    i += 1

ad_path = "/Users/admin/Documents/PFN/ml/problem3"

""" 敵対画像の生成 """
adv(path, ad_path, epzero, model_param, true)


#advarsarial imagesに対する予測性能を測る
acc = accuracy(ad_path, true, model_param)
acc_b = accuracy(ad_baseline, true, model_param)
print("advarasarial accuracy : " + str(acc))
print("baseline accuracy :" + str(acc_b))
