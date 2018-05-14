import math


def add(x,y):
    return list(map(lambda x,y: x+y, x,y))

#行列Aとベクトルxの掛け算
#行列はリストのリストで与えられる
def dot(A, x):
    #ベクトル間の内積計算
    def inner(x,y):
        s = 0
        for a,b in zip(x,y):
            s = s + (a*b)
        return s
    new_vec = []
    for vec in A:
        new_vec.append(inner(vec, x))
    return new_vec

#行列の転置
def trans(A):
    return list(map(list, zip(*A)))

#ベクトルの各要素をReLUに通す
def ReLU(x):
    return list(map((lambda x: x if(x>0) else 0), x))

def softmax(x):
    mom = 0
    for a in x:
        mom = mom + math.exp(a)
    v = []
    for e in x:
        v.append(math.exp(e)/mom)
    return v

""" One image -> vector """
def tovector(path):
    with open(path) as f:
         lines = f.read()[13:]
    v = []
    #valueは画素値
    for value in lines.split():
        v.append(int(value)/255)
    return v

""" get parameters of model from param.txt """
def get_model_param(path):
    with open(path) as f:
        lines = f.readlines()

    W1,W2,W3= [],[],[]
    b1,b2,b3 = 0,0,0

    for l in lines[:256]:
        W1.append(list(map(float, l.split())))
    for l in lines[257:513]:
        W2.append(list(map(float, l.split())))
    for l in lines[514:537]:
        W3.append(list(map(float, l.split())))
    b1 = list(map(float, lines[256].split()))
    b2 = list(map(float, lines[513].split()))
    b3 = list(map(float, lines[537].split()))

    return W1,W2,W3,b1,b2,b3

""" get true labels from labels.txt """
def get_true_label(path):
    with open(path) as f:
        line = f.read()
    true = line.split("\n")
    true = list(map(int, true[:-1]))
    return true

""" 画像書き込み """
""" img is a list of values of pixels of image. """
def write_image(img):
    head = "P2\n32 32\n255\n"
    j = 1
    for v in img:
        if(j % 32 == 0):
            head = head + str(v) + "\n"
        else:
            head = head + str(v) + " "
        j = j + 1
    return head
