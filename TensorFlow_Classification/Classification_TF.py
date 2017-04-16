# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:37:24 2017

@author: gaetano
"""

import tensorflow as tf
import numpy as np
import scipy as scp

# ==================== DATA IMPORT - Data normalization
#dataTrain = scp.io.loadmat("dataNorm.mat")["yNorm"]
#twoClasses = scp.io.loadmat("ClassificationTwoClasses.mat")["class_id2Classes"]
#fullClasses = scp.io.loadmat("ClassificationFullClasses.mat")["classIdOriginal"]
originalData = scp.io.loadmat("originalDataImport.mat")
originalData = originalData.get("arr")
nRows = originalData.shape[0]
nCols = originalData.shape[1]
twoClasses = originalData[:, nCols - 1]
twoClasses = twoClasses - 1
twoClasses = np.reshape(twoClasses, (nRows, 1))

originalData = np.delete(originalData, nCols - 1, 1)

media = np.mean(originalData, 0) * np.ones((1, nCols-1))
varianza = np.var(originalData, 0) * np.ones((1, nCols-1))
dataTrain = (originalData - media) / (np.sqrt(varianza))

# =========================== Neural network building 
nCols = nCols - 1
learningRate = 10e-5
HN_FirstLayer = 257
HN_SecondLayer = 128

xPlaceholder = tf.placeholder(tf.float32, shape = (nRows, nCols))
yPlaceholder = tf.placeholder(tf.float32, shape = (nRows, 1))

variables = []  # Contains variables list

# First layer
w1 = tf.Variable(tf.random_normal(shape = [nCols, HN_FirstLayer], mean = 0.0, \
                          stddev=1.0, dtype = tf.float32), name = "w1")
variables.append(w1)
b1 = tf.Variable(tf.random_normal(shape = [1, HN_FirstLayer], mean=0.0, \
                          stddev=1.0, dtype = tf.float32), name = "b1")
variables.append(b1)

# Second layer
w2 = tf.Variable(tf.random_normal(shape = [HN_FirstLayer, HN_SecondLayer], \
                          mean=0.0, stddev=1.0, dtype = tf.float32), name = "w2")
variables.append(w2)
b2 = tf.Variable(tf.random_normal(shape = [1, HN_SecondLayer], mean=0.0, \
                          stddev=1.0, dtype = tf.float32), name = "b2")
variables.append(b2)
# Output layer
w3 = tf.Variable(tf.random_normal(shape = [HN_SecondLayer, 1], mean=0.0, \
                          stddev=1.0, dtype = tf.float32), name = "w3")
variables.append(w3)
b3 = tf.Variable(tf.random_normal(shape = [1, 1]), name = "b3")
variables.append(b3)

a1 = tf.matmul(xPlaceholder, w1) + b1
z1 = tf.nn.sigmoid(a1)
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.sigmoid(a2)
estimationTwoClasses = tf.matmul(z2, w3) + b3

costFunction = tf.reduce_sum(tf.squared_difference(yPlaceholder, \
                                   estimationTwoClasses), name = "obj_funct")
optim = tf.train.GradientDescentOptimizer(learningRate, name = "GradDes")
optim_op = optim.minimize(costFunction, var_list = variables)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # --> Initializes all variables, BUT NOTHING ELSE!

for k in range(100000):
    xGen = dataTrain
    yGen = twoClasses
    trainData = {xPlaceholder:xGen, yPlaceholder:yGen}
    sess.run(optim_op, feed_dict = trainData)
    
output = estimationTwoClasses.eval(feed_dict = trainData, session = sess)
output = np.round(output)

truePositive = 1
trueNegative = 1
falsePositive = 1
falseNegative = 1
corrDet = 0

for h in range(nRows):
    if (output[h] == 1 and twoClasses[h] == 1):
        truePositive = truePositive + 1
    if (output[h] == 0 and twoClasses[h] == 0):
        trueNegative = trueNegative + 1
    if (output[h] == 1 and twoClasses[h] == 0):
        falsePositive = falsePositive + 1
    if (output[h] == 0 and twoClasses[h] == 1):
        falseNegative = falseNegative + 1
    if (output[h] == twoClasses[h]):
        corrDet = corrDet + 1

sensitivity = truePositive / (truePositive + falseNegative)
specificity = trueNegative / (trueNegative + falsePositive)
print("Sensitivity = " + str(sensitivity))
print("Specificity = " + str(specificity))
print("Correct detection = " + str(corrDet/nRows))