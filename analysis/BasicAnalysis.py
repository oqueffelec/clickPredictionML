import time

import numpy as np

from analysis.DataInstance import DataInstance
from analysis.DataSet import DataSet


class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    instance = dataset.nextInstance()
    while dataset.hasNext():
        y = instance.tokens
        # tokens uniques de la ligne i
        x = np.unique(y)
        instance = dataset.nextInstance()
    X = np.array(x)
    # tokens uniques de X = [x1' x2' ... xi' ... xn']'
    X = np.unique(X)

    return X

  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self, dataset):
    # TODO: Fill in your code here
    dico=dict()
    datainstance=DataInstance(dataset.file_handler.readline(),dataset.has_label)
    while dataset.hasNext():
        dico[datainstance.userid]=1+dico[datainstance.userid]
        dataset.nextInstance()
    list_uniq=set()
    indice=0
    for ids in dico.keys():
        if dico[ids]==0:
            list_uniq[indice]=ids
            indice=indice+1
    return list_uniq

  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self, dataset):
    # TODO: Fill in your code here
    return {}

  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self, dataset):
    f = open(dataset.path)
    ll = []
    for l in f:
      y = int(l[0])
      ll.append(y)
    x = np.array(ll)
    f.close()
    ctr = np.sum(x) / dataset.size
    return ctr

if __name__ == '__main__':

  TRAININGSIZE = 2335859
  fname = "/home/rasendrasoa/workspace/ClickPrediction/data/train.txt"
  training = DataSet(fname, True, TRAININGSIZE)
  analysis = BasicAnalysis()

  t1 = time.clock()
  ctr = analysis.average_ctr(training)
  t2 = time.clock()
  print("Average CTR = ",ctr*100,"%")
  print("temps de calcul pour la moyenne : ",t2-t1 ,'secondes')


  t1 = time.clock()
  X = analysis.uniq_tokens(training)
  t2 = time.clock()
  print('there are ',X.size,' unique tokens')
  print('temps de calcul unique tokens ', t2 - t1, "secondes")
