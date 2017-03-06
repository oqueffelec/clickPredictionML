import csv
from scipy import sparse
import mmap
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

#fonction pour le nombre de ligne du fichier
def mapcount(filename):
    with open(filename, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
    return lines

fname = "/home/oqueffelec/Documents/PAO/data/train.txt"
file = open(fname, "r")
reader = csv.reader(file, CPDialect())

t1 = time.clock()
nbLigne = mapcount(fname)
t2 = time.clock()
TnbLigne = t2-t1

X = 0

t3 = time.clock()
for row in reader:
    temp = int(row[0])
    X = temp + X
t4 = time.clock()
TSomme = t2-t1

print('nb de clicks :',X)
print('nbLigne :' ,nbLigne)

print('moyenne', X/nbLigne)

print('tps  calcul nb clicks en s :',TSomme)
print('tps calcul nb ligne : en s',TnbLigne)
file.close()
