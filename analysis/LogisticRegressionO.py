# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.sparse import lil_matrix

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil


# This class represents the weights in the logistic regression model.
class Weights:
    def __init__(self):
        self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
        # token feature weights
        self.w_tokens = np.zeros(shape=(1070659,1))
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
        l2 = self.w0 * self.w0 + \
             self.w_age * self.w_age + \
             self.w_gender * self.w_gender + \
             self.w_depth * self.w_depth + \
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
        w = [float(weights.w0), float(weights.w_age), float(weights.w_gender), float(weights.w_depth), float(weights.w_position)]
        x = [1.0, float(instance.age), float(instance.gender), float(instance.depth), float(instance.position)]
        temp=0
        for i in instance.tokens:
            temp+=weights.w_tokens[i]
        print(np.inner(w,x))+temp
        return np.inner(w,x)   + temp

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
        #weights = Weights()
        maxTokenValue = 1070659
        offset = 5
        weights= Weights()
        n_epoch = 1
        for epoch in range(n_epoch):
            while (dataset.hasNext()):
                instance = dataset.nextInstance()
                prediction = self.predict(weights, instance)
                print("prediction=", prediction)
                error =  instance.clicked - prediction
                weights.w0 = weights.w0 + step * error
                weights.w_age = weights.w_age + step * error * instance.age
                weights.w_gender = weights.w_gender + step * error * instance.gender
                weights.w_depth = weights.w_depth + step * error * instance.depth
                weights.w_position = weights.w_position + step * error * instance.position
                for indice in instance.tokens:
                    weights.w_tokens[indice]=weights.w_tokens[indice]+step*error

        print("train DONE")
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
    def predict(self, weights, instance):
        product = self.compute_weight_feature_product(weights, instance)
        activation = (math.exp(float(product)))/(1+(math.exp(float(product))))
        return activation

    def predictTest(self, weights, instance):
        teta=0
        product = self.compute_weight_feature_product(weights, instance)
        if (product>teta):
            return 1
        return 0


if __name__ == '__main__':
    # TODO: Fill in your code here
    fname = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/train.txt"
    TRAININGSIZE = 9000
    training = DataSet(fname, True, TRAININGSIZE)
    logisticregression = LogisticRegression()
    poids = logisticregression.train(training, 0, 0.1, 0)
    print(poids)
    fname = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/test.txt"
    TESTINGSIZE = 5000
    testing = DataSet(fname, False, TESTINGSIZE)
    res = np.empty(TESTINGSIZE)
    for i in range(TESTINGSIZE):
        instance = testing.nextInstance()
        res[i]=logisticregression.predictTest(poids,instance)
    print(np.where(res>0))
    fname = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/test_label.txt"
    testinglabel = DataSet(fname, True, TESTINGSIZE)
