#!/usr/bin/env python
"""
Monte Carlo simulation of the 2D Ising model
"""

from scipy import *
#from scipy import weave
from pylab import *

Nitt = 1000000  # total number of Monte Carlo steps
N = 10  # linear dimension of the lattice, lattice-size= N x N
warm = 1000  # Number of warmup steps
measure = 100  # How often to take a measurement


def CEnergy(latt):
    "Energy of a 2D Ising lattice at particular configuration"
    Ene = 0
    for i in range(len(latt)):
        for j in range(len(latt)):
            S = latt[i, j]
            WF = latt[(i + 1) % N, j] + latt[i, (j + 1) % N] + latt[(i - 1) % N, j] + latt[i, (j - 1) % N]
            Ene += -WF * S  # Each neighbor gives energy 1.0
    return Ene / 2.  # Each par counted twice


def RandomL(N):
    "Radom lattice, corresponding to infinite temerature"
    latt = zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            latt[i, j] = sign(2 * rand() - 1)
    return latt


def SamplePython(Nitt, latt, PW):
    "Monte Carlo sampling for the Ising model in Pythons"
    Ene = CEnergy(latt)  # Starting energy
    Mn = sum(latt)  # Starting magnetization

    Naver = 0  # Measurements
    Eaver = 0.0
    Maver = 0.0

    N2 = N * N
    for itt in range(Nitt):
        t = int(rand() * N2)
        (i, j) = (t % N, t / N)
        S = latt[i, j]
        WF = latt[(i + 1) % N, j] + latt[i, (j + 1) % N] + latt[(i - 1) % N, j] + latt[i, (j - 1) % N]
        P = PW[4 + S * WF]
        if P > rand():  # flip the spin
            latt[i, j] = -S
            Ene += 2 * S * WF
            Mn -= 2 * S

        if itt > warm and itt % measure == 0:
            Naver += 1
            Eaver += Ene
            Maver += Mn

    return (Maver / Naver, Eaver / Naver)

if __name__ == '__main__':

    latt = RandomL(N)
    PW = zeros(9, dtype=float)

    wT = linspace(4, 0.5, 100)
    wMag = []
    wEne = []
    wCv = []
    wChi = []
    for T in wT:
        # Precomputed exponents
        PW[4 + 4] = exp(-4. * 2 / T)
        PW[4 + 2] = exp(-2. * 2 / T)
        PW[4 + 0] = exp(0. * 2 / T)
        PW[4 - 2] = exp(2. * 2 / T)
        PW[4 - 4] = exp(4. * 2 / T)

        (maver, eaver) = SamplePython(Nitt, latt, PW)
        #(aM, aE, cv, chi) = SampleCPP(Nitt, latt, PW, T)
        wMag.append(aM / (N * N))
        wEne.append(aE / (N * N))
        wCv.append(cv / (N * N))
        wChi.append(chi / (N * N))

        print
        T, aM / (N * N), aE / (N * N), cv / (N * N), chi / (N * N)

    plot(wT, wEne, label='E(T)')
    plot(wT, wCv, label='cv(T)')
    plot(wT, wMag, label='M(T)')
    xlabel('T')
    legend(loc='best')
    show()
    plot(wT, wChi, label='chi(T)')
    xlabel('T')
    legend(loc='best')
    show()
