import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0,0,1])
x2 = np.array([0,1,1])
x3 = np.array([1,0,1])
x4 = np.array([1,1,1])

w = np.zeros((3))
x = np.array([x1,x2,x3,x4])
print('vecteur x1: ',np.shape(x1))
print('vecteur des poids',np.shape(w))
y = np.array([-1,1,1,1])
k = 67

l = np.size(x[:,1])
for count in range(1,k):
    indice = np.random.permutation(range(l))
    for i in range(1,l):
        i = indice[i]
        if(np.dot(y[i],np.inner(x[i,:],w))<=0):
            w += np.dot(y[i],x[i,:])


t = np.linspace(-5,5,10/0.1)
plt.plot(t,-(w[0]/w[1])*t-w[2]/w[1])
plt.ylabel("x2")
plt.xlabel("x1")
plt.hold(True)
for j in range(0,l):
    if(y[j]==1):
        plt.plot(x[j,0], x[j,1],'r+')
    else:
        plt.plot(x[j, 0], x[j, 1],'g*')
plt.hold(False)
plt.show()
