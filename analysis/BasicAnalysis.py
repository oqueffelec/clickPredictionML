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
    while dataset.hasNext():
      if dataset.nextInstance().clicked == 1:
        count += 1

    ctr = count/dataset.size
    return ctr

if __name__ == '__main__':
  TRAININGSIZE = 2335859
  training = DataSet("/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/train.txt", True, TRAININGSIZE)
  analysis = BasicAnalysis()

  # unique_users=analysis.uniq_users(training)
  # i=0
  # while i<400:
  #     print(list(unique_users)[i])
  #     i=i+1

  unique_users_age=analysis.uniq_users_per_age_group(training)
print(unique_users_age[0])
