import csv
from scipy import sparse
import numpy as np
import time

class CPDialect(csv.Dialect):
    # Séparateur de champ
    delimiter = "|"
    # Séparateur de ''chaîne''
    quotechar = None
    # Gestion du séparateur dans les ''chaînes''
    escapechar = None
    doublequote = None
    # Fin de ligne
    lineterminator = "\r\n"
    # Ajout automatique du séparateur de chaîne (pour ''writer'')
    quoting = csv.QUOTE_NONE
    # Ne pas ignorer les espaces entre le délimiteur de chaîne
    # et le texte
    skipinitialspace = False

def my_loadtxt(file):
    # barebones loadtxt
    f = open(file)
    h = f.readline()
    ll = []
    for l in f:
        y = int(l[0])
        ll.append(y)
    x = np.array(ll)
    f.close()
    return x

TRAININGSIZE = 2335859
fname = "/home/rasendrasoa/workspace/ClickPrediction/data/train.txt"
file = open(fname, "r")

rows, columns = TRAININGSIZE, 1
matrix = sparse.lil_matrix( (rows, columns) )
csvreader = csv.reader(file, CPDialect())

X = my_loadtxt(fname)
t1 = time.clock()
CTR = np.sum(X)/TRAININGSIZE
t2 = time.clock()

print("CTR = ",CTR," fait en ",t2-t1," secondes")