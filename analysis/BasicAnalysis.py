import time


import numpy as np

from analysis.DataInstance import DataInstance
from analysis.DataSet import DataSet


import csv
import numpy as np

class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    X =[]
    instance = dataset.nextInstance()
    while dataset.hasNext():
        y = instance.tokens
        # tokens uniques de la ligne i
        x = np.unique(y)
        X.append(x)
        instance = dataset.nextInstance()
    X = np.array(x)
    # tokens uniques de X = [x1' x2' ... xi' ... xn']'
    X = np.unique(X)

    return X
    # TODO: Fill in your code here
    return set()


  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self, dataset):
    #TODO: Fill in your code here
    datainstance=dataset.nextInstance()
    unique_users=set()
    while dataset.hasNext():
        unique_users.add(datainstance.userid)
        datainstance=dataset.nextInstance()
    return unique_users
  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self, dataset):
    # TODO: Fill in your code here
    datainstance=dataset.nextInstance()
    unique_users0=set()
    unique_users1=set()
    unique_users2=set()
    unique_users3=set()
    unique_users4=set()
    unique_users5=set()
    unique_users6=set()
    while dataset.hasNext():
        if datainstance.age==0:
            unique_users0.add(datainstance.userid)
        elif datainstance.age==1:
            unique_users1.add(datainstance.userid)
        elif datainstance.age==2:
            unique_users2.add(datainstance.userid)
        elif datainstance.age==3:
            unique_users3.add(datainstance.userid)
        elif datainstance.age==4:
            unique_users4.add(datainstance.userid)
        elif datainstance.age==5:
            unique_users5.add(datainstance.userid)
        else:
            unique_users6.add(datainstance.userid)
        datainstance=dataset.nextInstance()
    a=[unique_users0, unique_users1, unique_users2, unique_users3, unique_users4, unique_users5, unique_users6]
    return a
  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self, dataset):
    temp = 0
    count = 0
    x = []
    while dataset.hasNext():
      temp = dataset.nextInstance().clicked
      x.append(temp)
    return np.sum(x)/dataset.size

if __name__ == '__main__':

  TRAININGSIZE = 2335859

  fname = "/home/rasendrasoa/workspace/data/train.txt"
  training = DataSet(fname, True, TRAININGSIZE)
  analysis = BasicAnalysis()

  t1 = time.clock()
  #ctr = analysis.average_ctr(training)
  t2 = time.clock()
  #print("Average CTR = ",ctr*100,"%")
  print("temps de calcul pour la moyenne : ",t2-t1 ,'secondes')


  t1 = time.clock()
  X = analysis.uniq_tokens(training)
  t2 = time.clock()
  print('there are ',X.size,' unique tokens')
  print(X)
  print('temps de calcul unique tokens ', t2 - t1, "secondes")
  training = DataSet("/home/rasendrasoa/workspace/data/train.txt", True, TRAININGSIZE)
  analysis = BasicAnalysis()

  # unique_users=analysis.uniq_users(training)
  # i=0
  # while i<400:
  #     print(list(unique_users)[i])
  #     i=i+1

  unique_users_age=analysis.uniq_users_per_age_group(training)
print("nbre de unique tokens pour l'age 0 : ", len(unique_users_age[0]))
print("\n")
print("nbre de unique tokens pour l'age 1 : ", len(unique_users_age[1]))
print("\n")
print("nbre de unique tokens pour l'age 2 : ", len(unique_users_age[2]))
print("\n")
print("nbre de unique tokens pour l'age 3 : ",len(unique_users_age[3]))
print("\n")
print("nbre de unique tokens pour l'age 4 : " ,len(unique_users_age[4]))
print("\n")
print("nbre de unique tokens pour l'age 5 : ", len(unique_users_age[5]))
print("\n")
print("nbre de unique tokens pour l'age 6 : ", len(unique_users_age[6]))
print("\n")
