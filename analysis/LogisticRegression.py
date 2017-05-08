import math
import numpy as np

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

# This class represents the weights in the logistic regression model.
class Weights:
  def __init__(self):
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # token feature weights
    self.w_tokens = {}
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.
    self.access_time = {}

  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Tokens: " + str(self.w_tokens) + "\n"
    return string

  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_tokens.values():
      l2 += w * w
    return math.sqrt(l2)

  # @return {Int} the l2 norm of this weight vector
  def l0_norm(self):
    return 4 + len(self.w_tokens)


class LogisticRegression:
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):
    # TODO: Fill in your code here
    tab_itokens = np.asarray(instance.tokens)
    tab_wtokens = np.zeros(np.shape(tab_itokens))
    for v in weights.w_tokens.values():
        np.append(tab_wtokens,v)
    w = [weights.w0, weights.w_age, weights.w_gender, weights.w_depth, weights.w_position]
    vect_inst = [1, instance.age, instance.gender, instance.depth, instance.position]
    tab_itokens = np.asarray(instance.tokens)
    return np.inner(w,vect_inst)#+np.inner(tab_wtokens,tab_itokens)

  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param tokens {[Int]} list of tokens
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, tokens, weights, now, step, lambduh):
    # TODO: Fill in your code here
    return

  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, lambduh, step, avg_loss):
    weights = Weights()
    n_epoch=6
    for epoch in range(n_epoch):
      sum_error = 0.0
      while(dataset.hasNext()):
        prediction = self.predict(weights,dataset)
        instance=dataset.nextInstance()
        error = instance.clicked - prediction
        sum_error += error**2
        weights.w0 = weights.w0 + step * error
        weights.w_age = weights.w_age + step * error * instance.age
        weights.w_gender = weights.w_gender + step * error * instance.gender
        weights.w_depth = weights.w_depth + step * error * instance.depth
        weights.w_position = weights.w_position + step * error * instance.position
        for key,value in weights.w_tokens.items():
            weights.w_tokens[key] = value + step*error*instance.tokens[key]
    dataset.reset()
    return weights


    # TODO: Fill in your code here. The structure should look like:
    # For each data point:
      # Your code: perform delayed regularization

      # Your code: predict the label, record the loss

      # Your code: compute w0 + <w, x>, and gradient

      # Your code: update weights along the negative gradient


  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    instance=dataset.nextInstance()
    activation= self.compute_weight_feature_product(weights,instance)
    return 1.0 if activation >= 0.0 else 0.0



if __name__ == '__main__':
  # TODO: Fill in your code here
  fname = "C:\Data\data/train.txt"
  TRAININGSIZE=90000
  training = DataSet(fname, True, TRAININGSIZE)
  logisticregression=LogisticRegression()
  poids=logisticregression.train(training,0,0.1,0)
  print(poids)
  fname = "C:\Data/data/test.txt"
  TESTINGSIZE=500
  testing = DataSet(fname, False, TESTINGSIZE)
  res=np.empty(TESTINGSIZE)
  for k,v in poids.w_tokens.items():
    print(k,v)
  print(len(poids.w_tokens))