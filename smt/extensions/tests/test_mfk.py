# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:20:11 2018

@author: m.meliani
"""

import numpy as np
import matplotlib.pyplot as plt


def cheap(Xc):
    return 0.5*((Xc*6-2)**2)*np.sin((Xc*6-2)*2)+(Xc-0.5)*10. - 5

def expensive(Xe):
    return ((Xe*6-2)**2)*np.sin((Xe*6-2)*2)

dim = 1
Xe = np.linspace(0,1, 4).reshape(-1, dim)
Xc = np.linspace(0,1, 10).reshape(-1, dim)

n_doe = Xe.size
n_cheap = Xc.size

ye = expensive(Xe)
yc = cheap(Xc)

Xr = np.linspace(0,1, 100)
Yr = expensive (Xr)
from smt.extensions.mfkpls import MFKPLS

sm = MFKPLS(n_comp = 1,theta0=np.array(Xe.shape[1]*[1.]), model = 'KPLS', 
               eval_noise = False, print_global = False)

sm.set_training_values(Xc, yc, name= 0)
sm.set_training_values(Xe, ye)
sm.train() #low-fidelity dataset names being integers from 0 to level-1


x = np.linspace(0, 1, 101, endpoint = True).reshape(-1,1)
y= sm.predict_values(x)
MSE = sm.predict_variances(x)
der = sm.predict_derivatives(x, kx = 0)
ycheap = sm.LF_model.predict_values(x)
plt.figure()

plt.plot(Xr, expensive(Xr), label ='reference')
plt.plot(x, y, label ='mean_gp')
plt.plot(x, ycheap, label ='mean_gp_cheap')
plt.plot(Xe, ye, 'ko', label ='expensive doe')
plt.plot(Xc, yc, 'g*', label ='cheap doe')
plt.plot(x, der/20, label ='der')
plt.axhline(0, linestyle = '--')
plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(MSE)),np.ravel(y+3*np.sqrt(MSE)), facecolor ="lightgrey", edgecolor="g" ,label ='tolerance +/- 3*sigma')
plt.legend(loc=0)
plt.ylim(-10,17)
plt.xlim(-0.1,1.1)
plt.show()
