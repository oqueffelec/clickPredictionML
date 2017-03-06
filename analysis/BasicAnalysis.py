from scipy import sparse
from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from analysis.DataInstance import DataInstance
import time
import csv
import numpy as np


class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    # TODO: Fill in your code here
    return set()

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
    h = f.readline()
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
  training = DataSet("/home/rasendrasoa/workspace/ClickPrediction/data/train.txt", True, TRAININGSIZE)
  analysis = BasicAnalysis()
  ctr = analysis.average_ctr(training)

  t1 = time.clock()
  print("Average CTR = ",ctr)
  t2 = time.clock()
  print("temps de calcul pour la moyenne : ",t2-t1 ,'secondes')

