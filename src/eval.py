from model import Model
from lib import *
import os



""" 画像の入ったディレクトリへのpathを受け取り、予測精度を返す関数 """
# trueは正解ラベル
def accuracy(path, true, model_par):
    def sort(value):
        return int(value.split(".")[0])

    W1,W2,W3,b1,b2,b3 = model_par[0], model_par[1], model_par[2],model_par[3], model_par[4], model_par[5]

    """ sorted_nameは、[1.pgm, 2.pgm, ...] """
    sorted_name = sorted(os.listdir(path), key=sort)
    path_to_images = []
    for name in sorted_name:
        path_to_images.append(path + "/" + name)

    """ images -> vectors """
    vectors = list(map(tovector, path_to_images))

    score = 0
    i = 1
    print("start predicting ...")
    for f,t in zip(vectors, true):
        model = Model(W1, b1, W2, b2, W3, b3, f)
        pred = model.predict()
        print(str(i) + ".pgm : " + "pred " + str(pred) + " true " +str(t))
        if(pred == t):
            score += 1
        i += 1
    print("finish predicting ...")
    acc = score/len(true)
    return acc



def sort(value):
    return int(value.split(".")[0])

""" 敵対画像の生成 """
#pathは画像ディレクトリへのパス
def adv(ori_path, adv_path, epzero, model_par, true):
    names =  sorted(os.listdir(ori_path), key=sort)
    path_to_images = []
    for p in names:
        path_to_images.append(ori_path + "/" + p)
    features = list(map(tovector, path_to_images))

    W1,W2,W3,b1,b2,b3 = model_par[0], model_par[1], model_par[2],model_par[3], model_par[4], model_par[5]
    i = 1
    for f in features:
        model = Model(W1, b1, W2, b2, W3, b3, f)
        adv_x = Model.make_advarsarial(model, epzero, true[i-1])
        adv_img = list(map(lambda x:0 if(x<0) else min([255, int(x*230)]), adv_x))
        adv_img = list(map(int, adv_img))
        filename = adv_path + "/" + str(i) + ".pgm"
        f = open(filename, "w")
        f.write(write_image(adv_img))
        f.close()

        i += 1
    return 0
