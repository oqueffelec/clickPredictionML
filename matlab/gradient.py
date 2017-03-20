import numpy as np

#problème avec data du sujet
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
print(y.shape)

n = 0.001;
w_old = np.ones((10,1))
print("taille du vecteur w_old :",w_old.shape)

n = 10

for j in range(n):
    h = ( y[j] - x[:,j] * w_old ) * x[:,j]
    w_new = w_old - n*h
    w_old = w_new;

print(w_new.shape)
print(np.transpose(x[:,0]).shape)

# les prédictions

F = np.zeros((10,1))

for i in range(n):
    F[i] = np.dot(np.transpose(x[:,i]),w_new)
    print(F[i])
