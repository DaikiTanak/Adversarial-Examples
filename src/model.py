from lib import *

class Model():
    """ x is an input image """
    def __init__(self,W1, b1, W2, b2, W3, b3, x):
        self.dim = 23
        self.x = x
        self.W1, self.W2, self.W3 = W1,W2,W3
        self.b1, self.b2, self.b3 = b1,b2,b3
        self.a1 = add(dot(W1, x), b1)
        self.h1 = ReLU(self.a1)
        self.a2 = add(dot(W2, self.h1), b2)
        self.h2 = ReLU(self.a2)
        self.y = add(dot(W3, self.h2), b3)
        self.f_x = softmax(self.y)

    def predict(self):
        return self.f_x.index(max(self.f_x)) + 1

    """ 敵対画像を作る """
    #t is the true label. epzero is the parameter of FGSM.
    def make_advarsarial(self, epzero, t):
        def cross_entropy(x, f_x):
            L = -proba[t-1] + sum(list(map(math.exp, f_x)))
            return L

        #t番目に-１がたつdim次元ベクトル
        def one_hot(t, dim):
            v = []
            for i in range(dim):
                if(i == t-1):
                    v.append(-1)
                else:
                    v.append(0)
            return v

        def backward(P, Q):
            return list(map(lambda x,y: x if y > 0 else 0, P,Q))

        L_y = add(one_hot(t, 23), self.f_x)
        L_h2 = dot(trans(self.W3), L_y)
        L_a2 = backward(L_h2, self.a2)
        L_h1 = dot(trans(self.W2), L_a2)
        L_a1 = backward(L_h1, self.a1)
        L_x = dot(trans(self.W1), L_a1)

        sign_L_x = list(map(lambda x: 1 if x>0 else 0, L_x))
        ep = list(map(lambda x: x * epzero,  sign_L_x))

        return add(self.x, ep)

    """ baseline : ep is 1 or -1 rondomly """
    def make_advarsarial_bl(self, epzero):
        import random
        def rand():
            r = random.random()
            if(r <= 0.5):
                return -1
            else:
                return 1
        """ eは各要素が-1,+1のベクトル """
        e = list(map(lambda x: rand(), self.x))
        ep = list(map(lambda x: x * epzero, e))
        return add(self.x, ep)
