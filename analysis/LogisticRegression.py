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
    w=[weights.w0, weights.w_age, weights.w_gender, weights.w_depth, weights.w_position]
    vect_inst=[instance.clicked, instance.age, instance.gender, instance.depth, instance.position]
    return np.inner(w,vect_inst)

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
    w=[weights.w_age, weights.w_gender, weights.w_depth, weights.w_position,weights.w0]
    l = dataset.size
    k = 6
    for count in range(1,k):
        indice = np.random.permutation(range(l))
        for i in range(1,l):
            i = indice[i]
            instance = dataset.nextIemeInstance(i)
            x=[instance.age, instance.gender, instance.depth, instance.position,1]
            y=instance.clicked
            if(np.dot(y,np.inner(x,w))<=0):
                w += np.dot(y,x)


    # TODO: Fill in your code here. The structure should look like:
    # For each data point:
      # Your code: perform delayed regularization

      # Your code: predict the label, record the loss

      # Your code: compute w0 + <w, x>, and gradient

      # Your code: update weights along the negative gradient
    weights.w_age=w[0]
    weights.w_gender=w[1]
    weights.w_depth=w[2]
    weights.w_position=w[3]
    weights.w0=w[4]
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    activation = weights.w0
    W = [weights.w0, weights.w_age, weights.w_gender, weights.w_depth, weights.w_position, weights.w_tokens]
    for i in range(weights.l0_norm()):
      instance = dataset.nextInstance()
      activation += LogisticRegression().compute_weight_feature_product(weights,instance)
    return 1.0 if activation > 0.0 else 0.0


if __name__ == '__main__':
  # TODO: Fill in your code here
  fname = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/train.txt"
  TRAININGSIZE=900
  training = DataSet(fname, True, TRAININGSIZE)
  logisticregression=LogisticRegression()
  poids=logisticregression.train(training,0,0,0)
  print(poids)
  print(logisticregression.predict(poids,training))

  #
  # instance= training.nextInstance()
  # weights=Weights()
  # logisticregression=LogisticRegression()
  # prod_scal = logisticregression.compute_weight_feature_product(weights,instance)
  # print("Training Logistic Regression...")
  # print(prod_scal)
  # print("Training Logistic Regression...")
  #
  # F = logisticregression.predict(weights,training)
  # print("prediction :",F)
  # print(weights)
  #
