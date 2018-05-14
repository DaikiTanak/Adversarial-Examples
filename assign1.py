from lib import add,dot,trans,ReLU,softmax

x = [1,3]
y = [2,4]
A = [[1,2], [3,4]]

print("x + y")
print(add(x,y))
#this should be [3,7]


print("Ax")
print(dot(A,x))
#this should be [7,15]

print("A.T")
print(trans(A))
#[[1,3], [2,4]]


z = [-1,3,-5,7,2]

print("ReLU(z)")
print(ReLU(z))
#[0,3,0,7,2]

print("softmax(z)")
print(softmax(z))
