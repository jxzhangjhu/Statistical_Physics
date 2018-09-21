import numpy as np
import matplotlib.pyplot as plt

# testing by 09-18-2018
# 2D ising model

N = 10
shape = (N, N)

# Spin configuration
spins = np.random.choice([-1, 1], size=shape)

# Magnetic moment
moment = 1

# External magnetic field
field = np.full(shape, 0)

# Temperature (in units of energy)
temperature = 1

# Interaction (ferromagnetic if positive, antiferromagnetic if negative)
interaction = 1


def get_probability(energy1, energy2, temperature):
    return np.exp((energy1 - energy2) / temperature)


def get_energy(spins):
    return -np.sum(
        interaction * spins * np.roll(spins, 1, axis=0) +
        interaction * spins * np.roll(spins, -1, axis=0) +
        interaction * spins * np.roll(spins, 1, axis=1) +
        interaction * spins * np.roll(spins, -1, axis=1)
    ) / 2 - moment * np.sum(field * spins)


def update(spins, temperature):
    spins_new = np.copy(spins)
    i = np.random.randint(spins.shape[0])
    j = np.random.randint(spins.shape[1])
    spins_new[i, j] *= -1

    current_energy = get_energy(spins)
    new_energy = get_energy(spins_new)
    if get_probability(current_energy, new_energy, temperature) > np.random.random():
        return spins_new
    else:
        return spins

# show the initilization figure
im = plt.imshow(spins, cmap='gray', vmin=-1, vmax=1, interpolation='none')
plt.ion()
plt.show()

t = 0
# training datasets
N_train = 1000
xtrain = np.zeros((N_train, N*N))
ytrain = np.zeros((N_train, 1))

#while True:
for t in range(N_train):
    # if t % 100 == 0:
    #     im.set_data(spins)
    #     plt.draw()

    spins = update(spins, temperature)
    xtrain[t, :] = np.reshape(spins, (1, N*N))
    # xtrain.append(spins)
    # print(spins)
    Energy = get_energy(spins)
    ytrain[t] = Energy
    # ytrain.append(Energy)
    # print(Energy)
    # plt.pause(.001)
    t += 1
    # plt.show()

im.set_data(spins)
plt.draw()
plt.show()

print(ytrain)
print(len(ytrain))
print(xtrain.shape)
# plt.close()
plt.ioff()

plt.figure()
# plot the histogram
plt.hist(ytrain, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram")
plt.show()

##################################################################
# training with all the training data, K-fold
#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = xtrain
Y = ytrain


# define base Neural Network model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=N*N, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold, n_jobs=1)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

## Modeling the standardized datasets
# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

## Evaluate a deeper network topology
# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

## Evaluate a wider network topolgy
# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# # testing datasets
# t = 0
# # training datasets
# N_test = int(N_train*0.2)
# # xtest = []
# # ytest = []
# xtest = np.zeros((N_test, N*N))
# ytest = np.zeros((N_test, 1))
#
# #while True:
# for t in range(N_test):
#     # if t % 100 == 0:
#     #     im.set_data(spins)
#     #     plt.draw()
#     spins = update(spins, temperature)
#     xtest[t, :] = np.reshape(spins, (1, N*N))
#     # xtrain.append(spins)
#     # print(spins)
#     Energy = get_energy(spins)
#     ytest[t] = Energy
#     # print(Energy)
#     # plt.pause(.001)
#     t += 1
#     # plt.show()


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
# xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=7)
#
# xgb.fit(xtrain, ytrain)
# predictions = xgb.predict(xtest)
# print(xtrain)
# print(ytrain)
# print(xtest)
# print(ytest)
# print(explained_variance_score(predictions, ytest))