import sqlite3
# on va se connecter à une base de données SQL vide
# SQLite stocke la BDD dans un simple fichier
filepath = "./DataBase.db"
open(filepath, 'w').close() #crée un fichier vide
CreateDataBase = sqlite3.connect(filepath)

QueryCurs = CreateDataBase.cursor()

# On définit une fonction de création de table
def CreateTable(nom_bdd):
    QueryCurs.execute('''CREATE TABLE IF NOT EXISTS ''' + nom_bdd + '''
    (rowid INTEGER PRIMARY KEY AUTOINCREMENT,Clicked INTEGER, Depth INTEGER,Position INTEGER, Userid INTEGER, Age INTEGER,
    Gender REAL, TextTokens TEXT)''')

# On définit une fonction qui permet d'ajouter des observations dans la table
def AddEntry(nom_bdd, Clicked,Depth,Position,Userid,Age,Gender,TextTokens):
    QueryCurs.execute('''INSERT INTO ''' + nom_bdd + '''
    (Clicked,Depth,Position,Userid,Age,Gender,TextTokens) VALUES (?,?,?,?,?,?,?)''',(Clicked,Depth,Position,Userid,Age,Gender,TextTokens))

def importData(nom_fichier, nom_table):
    QueryCurs.execute('''INSERT INTO ''' + nom_table + '''
        (Clicked,Depth,Position,Userid,Age,Gender,TextTokens) VALUES ( ''' + nom_fichier + ''',readfile(''' + nom_fichier +
        '''))''')

# QueryCurs.execute(''' LOAD DATA INFILE ''' + nom_fichier + ''' INTO TABLE ''' + nom_table + ''' FIELDS TERMINATED BY '|' ''')
CreateTable('toto')

importData('test.txt', 'toto')

CreateDataBase.commit()
