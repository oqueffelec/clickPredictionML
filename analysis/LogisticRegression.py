import math
import time
import numpy as np
from scipy.sparse import lil_matrix

from analysis.DataSet import DataSet


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
        (features,index) = self.featuresArray(instance)
        w = np.array([weights.w0,weights.w_age,weights.w_gender,weights.w_depth,weights.w_position])
        a = w.T.dot(features[0:5])
        b = sum(features[5:len(index)])
        return a+b

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
        maxTokenValue = 1070659
        offset = 5
        w = lil_matrix((maxTokenValue + offset + 1, 1))
        x = lil_matrix((maxTokenValue + offset + 1, 1))
        n_epoch = 1
        for epoch in range(n_epoch):
            sum_error = 0.0
            while (dataset.hasNext()):
                prediction = self.predict(weights,dataset)
                if (1-prediction>=0.1):
                    print(prediction)
                instance = dataset.nextInstance()
                (features,index) = self.featuresArray(instance)
                error = instance.clicked - prediction
                grad = error*step
                x[index] *= grad
                w[index] += x[index]
                sum_error += error ** 2
                self.sparseToWeights(w,weights,index)
        dataset.reset()
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
    def predict(self,weights, dataset):
        instance = dataset.nextInstance()
        activation = self.compute_weight_feature_product(weights, instance)
        activation = math.exp(activation)
        activation = activation/(1+activation)
        return activation

    # Create features arrayones x and index for non-0 values
    def featuresArray(self, instance):
        temp = np.ones((len(instance.tokens)))
        features = np.array(
            [1.0, float(instance.age), float(instance.gender), float(instance.depth), float(instance.position)])
        features = np.concatenate((features, temp), axis=0)
        index = np.array([0, 1, 2, 3, 4])
        for i in range(len(instance.tokens)):
            instance.tokens[i] += 5
        temp = np.asarray(instance.tokens)
        index = np.concatenate((index, temp),axis =0)
        return (features, index)

    #return sparseVector
    def featureVector(self,x,instance):
        (features,index) = self.featuresArray(instance)
        for i in range(features.size):
            x[index[i]]=features[i]
        return x

    def sparseToWeights(self,w,weights,index):
        weights.w0 = w[0,0]
        weights.w_age = w[1,0]
        weights.w_gender = w[2,0]
        weights.w_depth = w[3,0]
        weights.w_position = w[4,0]
        for i in(index[5::]):
            weights.w_tokens[i] = w[i]
        #for i in(w.tocsc().indices):
            #weights.w_tokens.add(i,w[i+5])

if __name__ == '__main__':
    # TODO: Fill in your code here
    fname = "/home/rasendrasoa/workspace/data/train.txt"
    TRAININGSIZE = 10000
    training = DataSet(fname, True, TRAININGSIZE)
    logisticregression = LogisticRegression()
    t1 = time.clock()
    poids = logisticregression.train(training, 0, 0.1, 0)
    t2 = time.clock()
    print('train fait en',t2-t1,'s pour',TRAININGSIZE,'valeurs')
    fname = "/home/rasendrasoa/workspace/data/test.txt"
    TESTINGSIZE = 500
    testing = DataSet(fname, False, TESTINGSIZE)
    res = np.empty(TESTINGSIZE)
