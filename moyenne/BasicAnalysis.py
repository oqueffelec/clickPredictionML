from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
import time

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
    datainstance=DataInstance(self.file_handler.readline(),self.has_label)
    while dataset.hasNext():
        dico[datainstance.field[4]]=1+dico[datainstance.field[4]]
        dataset.nextInstance()
    list_uniq=set()
    indice=0
    for ids in dico.keys():
        if dico[ids]==0:
            list_uniq[indice]=ids
            indice=indice+1
    return set()

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
    temp = 0
    count = 0
    while dataset.hasNext():
      if dataset.nextInstance().clicked == 1:
        count += 1

    ctr = count/dataset.size
    return ctr

if __name__ == '__main__':
  TRAININGSIZE = 2335859
  training = DataSet("/home/rasendrasoa/workspace/ClickPrediction/data/train.txt", True, TRAININGSIZE)
  analysis = BasicAnalysis()
  t1 = time.clock()
  print(analysis.average_ctr(training))
  t2 = time.clock()
  print("temps de calcul pour la moyenne : ",t2-t1 ,'secondes')
