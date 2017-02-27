import mysql.connector
# on va se connecter à une base de données MySql
conn = mysql.connector.connect(host="localhost",user="root",password="sasa", database="test1")
cursor = conn.cursor()


def createTable(nom_bdd):
    cursor.execute('''CREATE TABLE IF NOT EXISTS ''' + nom_bdd + '''
    (rowid INTEGER PRIMARY KEY AUTOINCREMENT,Clicked INTEGER, Depth INTEGER,Position INTEGER, Userid INTEGER, Age INTEGER,
    Gender REAL, TextTokens TEXT)''')

def importData(nom_fichier, nom_table):
    cursor.execute(''' LOAD DATA INFILE ''' + nom_fichier + ''' INTO TABLE '''
                   + nom_table + ''' FIELDS TERMINATED BY '|' ''')

def moyenne(nom_table):
    cursor.execute(''' SELECT AVG(Clicked) FROM '''+ nom_table +'''; ''')

createTable('warmup')

importData('test.txt', 'toto')

moyenne(toto)

conn.close()