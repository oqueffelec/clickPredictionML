import numpy as np

#problème avec data duu sujet
#exemple pour valeurs random

x1 = np.random.rand(10,1)
x2 = np.random.rand(10,1)
x3 = np.random.rand(10,1)
x4 = np.random.rand(10,1)
x5 = np.random.rand(10,1)
x6 = np.random.rand(10,1)
x7 = np.random.rand(10,1)
x8 = np.random.rand(10,1)
x9 = np.random.rand(10,1)
x10 = np.random.rand(10,1)

x = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])
y = np.array([0,0,1,0,1,1,1,1,1,0])

print(x.shape)
print(y[1])
print(x[1])

n = 0.001;
w_old = np.ones((10,1))

n = 10

for j in range(n):
    h=(y[j]-x[:,j].T*w_old)*x[:,j]
    w_new=w_old-h*n;
    w_old=w_new;

F = []
for i in range(n):
    F[i] = x[i].T*w_new
