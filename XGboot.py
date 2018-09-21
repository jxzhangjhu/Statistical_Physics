from scipy import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

def RandomL(N):
    "Generates a random 2D Ising lattice"
# Randomly choosen <--> infinite temperature
    latt = zeros((N,N),dtype=int)
    for i in range(N):
        for j in range(N):
            latt[i,j] = sign(2*rand()-1)
    return latt

def CEnergy(latt):
    "Energy of a 2D Ising lattice"
    Ene = 0
    for i in range(N):
        for j in range(N):
            S = latt[i,j]
            WF = latt[(i+1)%N, j] + latt[i,(j+1)%N] + latt[(i-1)%N,j] + latt[i,(j-1)%N]
            Ene += -WF*S # Each neighbor gives energy 1.0
    return int(Ene/2.)   # Each par counted twice

## training
N = 10
N_train = 1000
xtrain = np.zeros((N_train, N*N))
ytrain = np.zeros((N_train, 1))

for i in range(N_train):
    latt = RandomL(N)
    xtrain[i, :] = np.reshape(latt, (1, N * N))
    Energy = CEnergy(latt)
    ytrain[i] = Energy

plt.hist(ytrain, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

# print(xtrain)
# print(xtrain.shape)
# print(ytrain)
# testing

N_test = int(N_train*0.2)
xtest = np.zeros((N_test, N*N))
ytest = np.zeros((N_test, 1))

for i in range(N_test):
    latt = RandomL(N)
    xtest[i, :] = np.reshape(latt, (1, N * N))
    Energy = CEnergy(latt)
    ytest[i] = Energy


# print(xtest)
# print(xtest.shape)
# print(ytest)


# # XGboot ML
# import xgboost
# from sklearn.metrics import explained_variance_score
# # print(len(xtrain))
# # param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# # num_round = 2
# # bst = xgb.train(param, xtrain, num_round)
# # # make prediction
# # preds = bst.predict(xtest)
#
# xgb = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=7)
#
# xgb.fit(xtrain, ytrain)
# preds = xgb.predict(xtest)
# print(preds)
# print(ytest)
# print(explained_variance_score(preds, ytest))