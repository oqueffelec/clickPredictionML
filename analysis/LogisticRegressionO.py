# -*- coding: utf-8 -*-
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

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
        for w in self.w_tokens:
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
        # il est indiqué dans l'énoncé que l'on peut considérer
        # la liste des tokens comme étant un vecteur creux b tel que b[i] = 1
        for i in instance.tokens:
            temp+=weights.w_tokens[i]
        return float(np.inner(w,x)) + temp

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
    def split(self, dataset, size):
        clicked=[]
        unclicked=[]
        i=0
        j=0
        instance=dataset.nextInstance()
        while(i<size):
            if(instance.clicked==0):
                instance=dataset.nextInstance()
            else:
                clicked.append(instance)
                instance=dataset.nextInstance()
                i+=1
        while(j<size):
            if(instance.clicked==1):
                instance=dataset.nextInstance()
            else:
                unclicked.append(instance)
                instance=dataset.nextInstance()
                j+=1
        return clicked,unclicked

    def regularised_train(self, dataset, lambduh, step, avg_loss):
        N = dataset.size/200
        weights= Weights()
        n_epoch = 1
        N00=0
        N01=0
        N10=0
        N11=0
        mse=[]
        count = 0
        nbStep = 100
        loss=0
        T = np.linspace(0,N,N/nbStep)
        clicked,unclicked=self.split(dataset,N)
        for epoch in range(n_epoch):
            for i in range(N):
                r=random.randint(0,1)
                j=random.randint(0,N-1)
                if(r==0):
                    instance = clicked[j]
                else:
                    instance = unclicked[j]
                prediction = self.predict(weights, instance)
                error =  instance.clicked - prediction
                if(error==0):
                    if(prediction==0):
                        N00+=1
                    else:
                        N11+=1
                else:
                    if(prediction==0):
                        N10+=1
                    else:
                        N01+=1
                # if (error!=0):
                weights.w0 = (1-step*lambduh/N)*weights.w0 + step * error
                weights.w_age = (1-step*lambduh/N)*weights.w_age + step * error * instance.age
                weights.w_gender = (1-step*lambduh/N)*weights.w_gender + step * error * instance.gender
                weights.w_depth = (1-step*lambduh/N)*weights.w_depth + step * error * instance.depth
                weights.w_position = (1-step*lambduh/N)*weights.w_position + step * error * instance.position
                for indice in instance.tokens:
                    weights.w_tokens[indice]= (1-step*lambduh/N)*weights.w_tokens[indice]+step*error
                # record the average loss for each step 100
                loss = loss + 10*(error * error)
                j = count % nbStep
                if (j == 0 and count / nbStep != 0 and count!=0):
                    avg_loss[int(count / nbStep)] = (1000000 / (2 * count)) * loss
                    print(avg_loss[int(count / nbStep)])
                count += 1

        print("train DONE")
        return weights,N00,N10,N01,N11,T,avg_loss


    def train(self, dataset, lambduh, step, avg_loss):
        N = dataset.size
        weights= Weights()
        n_epoch = 1
        N00=0
        N01=0
        N10=0
        N11=0
        count = 0
        nbStep = 100
        T = np.linspace(0,N,N/nbStep)
        for epoch in range(n_epoch):
            while (dataset.hasNext()):
                instance = dataset.nextInstance()
                prediction = self.predict(weights, instance)
                error =  instance.clicked - prediction
                if(error==0):
                    if(prediction==0):
                        N00+=1
                    else:
                        N11+=1
                else:
                    if(prediction==0):
                        N10+=1
                    else:
                        N01+=1
                # if (error!=0):
                weights.w0 = weights.w0 + step * error
                weights.w_age = weights.w_age + step * error * instance.age
                weights.w_gender = weights.w_gender + step * error * instance.gender
                weights.w_depth = weights.w_depth + step * error * instance.depth
                weights.w_position = weights.w_position + step * error * instance.position
                for indice in instance.tokens:
                    weights.w_tokens[indice]= weights.w_tokens[indice]+step*error
                # record the average loss for each step 100
                avg_loss[0] = (1 / 2) * (error * error)
                j = count % nbStep
                if (j == 0 and count / nbStep != 0):
                    avg_loss[int(count / nbStep)] = (1 / (2 * count)) * (error * error) + avg_loss[int(count / nbStep) - 1]
                count += 1

        print("train DONE")
        return weights,N00,N10,N01,N11,T,avg_loss

    # ==========================
    # Use the weights to predict the CTR for a dataset.
    # @param weights {Weights}
    # @param dataset {DataSet}
    # ==========================
    def predict(self, weights, instance):
        product = self.compute_weight_feature_product(weights, instance)
        if(product>0):
            return 1
        else:
            return 0

    # return a vector from values in file test_label.txt
    # @param fname {String}
    def test_labelTomatrix(self, fname):
        f = open(fname,"r")
        listeLabel = []
        count = 0
        reader = csv.reader(f)
        for row in reader:
            count+=1
            if(row[0]!='0'):
                temp = count
                listeLabel.append(temp)
        x = np.asarray(listeLabel)
        return x

    # calcule la matrice de confusion
    # @param dataset {DataSet}
    # @param tabLabel {array[TESTINGSIZE]}
    # @param TESTINGSIZE {Int}
    def test(self,dataset,tabLabel,TESTINGSIZE,poids):
        print('\n')
        print("TESTING ...")
        M00=0
        M01=0
        M10=0
        M11=0
        prediction = np.empty(TESTINGSIZE)
        for i in range(TESTINGSIZE):
            instance = dataset.nextInstance()
            prediction[i]=self.predict(poids,instance)
            if(i in tabLabel):
                if(prediction[i]==1):
                    M11+=1
                else:
                    M10+=1
            else:
                if(prediction[i]==0):
                    M00+=1
                else:
                    M01+=1
        return M00,M10,M01,M11



    def logreg(self, dataset, lambduh, step, avg_loss):

        N = dataset.size
        weights= Weights()
        n_epoch = 1
        N00=0
        N01=0
        N10=0
        N11=0
        count = 0
        nbStep = 100
        T = np.linspace(0,N,N/nbStep)
        for epoch in range(n_epoch):
            while (dataset.hasNext()):
                instance = dataset.nextInstance()
                prediction =self.sigmoid(self.compute_weight_feature_product(weights,instance))
                bayes=prediction/(1-prediction)
                error =  instance.clicked - prediction
                if(instance.clicked==0):
                    if(bayes<1):
                        N00+=1
                    else:
                        N01+=1
                else:
                    if(bayes<1):
                        N10+=1
                    else:
                        N11+=1
                # if (error!=0):
                weights.w0 = (1-step*lambduh/N)*weights.w0 + step * error
                weights.w_age = (1-step*lambduh/N)*weights.w_age + step * error * instance.age
                weights.w_gender = (1-step*lambduh/N)*weights.w_gender + step * error * instance.gender
                weights.w_depth = (1-step*lambduh/N)*weights.w_depth + step * error * instance.depth
                weights.w_position = (1-step*lambduh/N)*weights.w_position + step * error * instance.position
                for indice in instance.tokens:
                    weights.w_tokens[indice]= (1-step*lambduh/N)*weights.w_tokens[indice]+step*error
                # record the average loss for each step 100
                avg_loss[0] = (1 / 2) * (error * error)
                j = count % nbStep
                if (j == 0 and count / nbStep != 0):
                    avg_loss[int(count / nbStep)] = (1 / (2 * count)) * (error * error) + avg_loss[int(count / nbStep) - 1]
                count += 1

        print("train DONE")
        return weights,N00,N10,N01,N11,T,avg_loss


    def sigmoid(self,z):
        return 1/(1+math.exp(-1*z))




if __name__ == '__main__':
    # TODO: Fill in your code here
    train = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/train.txt"
    test_label = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/test_label.txt"
    test = "/Users/Octave/Documents/ASIBIS/gitPAO/clicks_prediction/data/test.txt"
    TRAININGSIZE = 2000000
    TESTINGSIZE = 5000
    lambduh=[0.01]
    step=0.01

    # # perform simple training
    # norm=[]
    # for l in lambduh:
    #     avg_loss = np.zeros((int(TRAININGSIZE/100),1))
    #     T = []
    #     training = DataSet(train, True, TRAININGSIZE)
    #     logisticregression = LogisticRegression()
    #     print('\n')
    #     print('\n')
    #
    #     print('Performing Simple Training...')
    #     poids,N00,N10,N01,N11,T,avg_loss = logisticregression.train(training, l, step, avg_loss)
    #     norm.append(poids.l2_norm())
    #     print('TRAININGSIZE =', TRAININGSIZE)
    #     print("LAMBDA = ",l)
    #     print("STEP = ",step)
    #     print(poids)
    #     print("N00",N00,"N10",N10) # N10 est un click qui est predis unclicked
    #     print("N01",N01,"N11",N11) # N01 est un not click qui est predis click
    #     print("Ratio de reussite pour le training",(N00+N11)/float(N00+N01+N10+N11))
    #     print("Average ctr for training ",(N10+N11)/float(N00+N01+N10+N11))
    #     print("Average ctr for training predicted",(N01+N11)/float(N00+N01+N10+N11))
    #
    # # perform logisticregression training
    # norm=[]
    # for l in lambduh:
    #     avg_loss = np.zeros((int(TRAININGSIZE/100),1))
    #     T = []
    #     training = DataSet(train, True, TRAININGSIZE)
    #     logisticregression = LogisticRegression()
    #     print('\n')
    #     print('\n')
    #
    #     print('Performing logreg Training...')
    #     poids,N00,N10,N01,N11,T,avg_loss = logisticregression.logreg(training, l, step, avg_loss)
    #     norm.append(poids.l2_norm())
    #     print('TRAININGSIZE =', TRAININGSIZE)
    #     print("LAMBDA = ",l)
    #     print("STEP = ",step)
    #     print(poids)
    #     print("N00",N00,"N10",N10) # N10 est un click qui est predis unclicked
    #     print("N01",N01,"N11",N11) # N01 est un not click qui est predis click
    #     print("Ratio de reussite pour le training",(N00+N11)/float(N00+N01+N10+N11))
    #     print("Average ctr for training ",(N10+N11)/float(N00+N01+N10+N11))
    #     print("Average ctr for training predicted",(N01+N11)/float(N00+N01+N10+N11))

    # perform regularised training
    norm=[]
    for l in lambduh:
        avg_loss = np.zeros((int(TRAININGSIZE/100),1))
        T = []
        training = DataSet(train, True, TRAININGSIZE)
        logisticregression = LogisticRegression()
        print('\n')
        print('\n')

        print('Performing Regularised Training...')
        poids,N00,N10,N01,N11,T,avg_loss = logisticregression.regularised_train(training, l, step, avg_loss)
        norm.append(poids.l2_norm())
        print('TRAININGSIZE =', TRAININGSIZE/200)
        print("LAMBDA = ",l)
        print("STEP = ",step)
        print(poids)
        print("N00",N00,"N10",N10) # N10 est un click qui est predis unclicked
        print("N01",N01,"N11",N11) # N01 est un not click qui est predis click
        print("Ratio de reussite pour le training",(N00+N11)/float(N00+N01+N10+N11))
        print("Average ctr for training ",(N10+N11)/float(N00+N01+N10+N11))
        print("Average ctr for training predicted",(N01+N11)/float(N00+N01+N10+N11))


    # # testing on test data set
    # tabLabel = logisticregression.test_labelTomatrix(test_label)
    # testing = DataSet(test, False, TESTINGSIZE)
    # M00,M10,M01,M11 = logisticregression.test(testing, tabLabel,TESTINGSIZE,poids)
    #
    # print("M00",M00,"M10",M01)
    # print("M01",M10,"M11",M11)
    # print("Ratio de reussite pour le testing",(M00+M11)/float(M00+M01+M10+M11))
    # print("Average ctr for testing",(M10+M11)/float(M00+M01+M10+M11))
    # print("Average ctr for testing predicted",(M01+M11)/float(M00+M01+M10+M11))
    #
    #
    # plt.figure(1)
    # plt.plot(lambduh, norm)
    # plt.title('l2 norm as a function of lambduh')
    # plt.ylabel('l2 norm')
    # plt.xlabel('lambduh')
    #
    #
    plt.figure(2)
    plt.plot(avg_loss)
    plt.title('Average Loss in function of the number of steps T')
    plt.ylabel('average loss')
    plt.xlabel('step = 100')
    plt.show()
