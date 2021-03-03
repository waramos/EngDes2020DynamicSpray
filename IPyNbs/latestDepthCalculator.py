# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:51:43 2020

@author: Saman Taba
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import pandas as pd
import timeit

PositionData = pd.read_excel('SphereR20Res05.xlsx', sheet_name='SphereR20Res05', header=0)
PosX = PositionData['X ']
PosY = PositionData[' Y ']
PosZ = PositionData[' Z']
#%%

def IndexMatcher(arr1, arr2):
    Index = []
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            if np.round(arr1[i],3) == np.round(arr2[j],3):
                Index.append(arr1[i])
    return Index

#%%
PosY = np.round(PosY, 3)
PosX = np.round(PosX, 3)

A = np.where(PosX == -5)[0]
B = np.where(PosY == -3.5)[0]
print(IndexMatcher(A, B))
#print(A)
#print(B)
#%%
start_time = timeit.default_timer()

step = 0.5
X = np.arange(-40, 40 + step, step)
Y = np.arange(-40, 40 + step, step)
Depth = np.zeros([len(X)*len(Y), 3])


Length = len(X)
c = 0
for i in range(Length):
    IndX = np.where(PosX == np.round( X[i],4))[0]
    for j in range(Length):
        IndY = np.where(PosY == np.round( Y[j],4))[0]
        if len(IndX) != 0 and len(IndY) != 0:
            N = IndexMatcher(IndX, IndY)
            if len(N) == 2:
                A = N[0]
                B = N[1]
                Thickness = np.abs(PosZ[A] - PosZ[B])
            elif len(N) == 4:
                A = N[0]
                B = N[1]
                C = N[2]
                D = N[3]
                
                Thickness = np.abs(PosZ[A] - PosZ[B]) + np.abs(PosZ[C] - PosZ[D]) 
            else:
                Thickness = 0
        else:
            Thickness = 0
        
        Depth[c, 0] = X[i]
        Depth[c, 1] = Y[j]
        Depth[c, 2] = Thickness
        c = c + 1

elapsed = timeit.default_timer() - start_time
print(elapsed)
#For sphere1, time is 11.3437822 s
#For sphere2, time is 34.677015 s
#For sphere3, time is 84.3042992 s
#For sphere4, time is 177.2984644 s
#For sphere5, time is 1974.6914911 s
#for radius 10 res= 0.2, time is 1519.97 s 
#%%
DepthX = Depth[:,0]
DepthY = Depth[:,1]
DepthZ = Depth[:,2]

x, y = np.meshgrid(X, Y)

ThicknessProfileT = np.zeros([len(Y), len(X)])
c = 0
for i in range(len(X)):
    for j in range(len(Y)):
        ThicknessProfileT[i, j] = DepthZ[c]
        c = c+1
#%%
Z = np.transpose(ThicknessProfileT)
plt.figure()
plt.contourf(x, y, Z, 20, cmap='RdGy', alpha=0.8)
#plt.colorbar();
#plt.contour(x, y, Z,  colors='black')
#plt.imshow(Z, cmap='RdGy', alpha=0.5)
cbar = plt.colorbar()
cbar.set_label('Thickness (microns)')
plt.xlabel('X Coordinate (microns)')
plt.ylabel('Y Coordinate (microns)')
plt.tight_layout()
plt.show()

Zt = 231 - Z
#%%
t = np.array([11.3437822, 34.677015, 84.3042992, 177.2984644, 573.705 ,1519.97])
t = t/60
size = np.array([2371, 3878, 5675, 7803, 15631, 23123])
plt.figure()
plt.plot(size, t, '-ko')
plt.ylabel("time of depth calcualtion (min)")
plt.xlabel("Number of data points")
plt.tight_layout()
plt.show()

'''
Length = len(PosX)
X = []
Y = []
Z = []

for i in range(Length):
    for j in range(i+1,Length):
        if PosX[i] == PosX[j] and PosY[i] == PosY[j]:
            depth = np.abs(PosZ[i] - PosZ[j])
            X.append(PosX[i])
            Y.append(PosY[i])
            Z.append(depth)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X,Y,Z, s=2)
'''