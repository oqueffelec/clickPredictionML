from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

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
      count += 1
      temp = dataset.nextInstance().clicked + temp
      ctr = temp/count
    return ctr

if __name__ == '__main__':
  TRAININGSIZE = 2335859
  training = DataSet("/home/rasendrasoa/workspace/ClickPrediction/data/train.txt", True, TRAININGSIZE)
  analysis = BasicAnalysis()
  print(analysis.average_ctr(training))
  print("Basic Analysis...")