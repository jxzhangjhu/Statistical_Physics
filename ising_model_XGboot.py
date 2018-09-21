import numpy as np
import matplotlib.pyplot as plt

# testing by 09-18-2018
# 2D ising model

N = 20
shape = (N, N)

# Spin configuration
spins = np.random.choice([-1, 1], size=shape)

# Magnetic moment
moment = 1

# External magnetic field
field = np.full(shape, 0)

# Temperature (in units of energy)
temperature = 0.1

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


plt.ion()
plt.show()

im = plt.imshow(spins, cmap='gray', vmin=-1, vmax=1, interpolation='none')
t = 0
# training datasets
N_train = 10000
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

# testing datasets
t = 0
# training datasets
N_test = int(N_train*0.2)
# xtest = []
# ytest = []
xtest = np.zeros((N_test, N*N))
ytest = np.zeros((N_test, 1))

#while True:
for t in range(N_test):
    # if t % 100 == 0:
    #     im.set_data(spins)
    #     plt.draw()
    spins = update(spins, temperature)
    xtest[t, :] = np.reshape(spins, (1, N*N))
    # xtrain.append(spins)
    # print(spins)
    Energy = get_energy(spins)
    ytest[t] = Energy
    # print(Energy)
    # plt.pause(.001)
    t += 1
    # plt.show()


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