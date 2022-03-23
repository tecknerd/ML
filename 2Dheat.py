# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:54:19 2021

@author: 18133
"""
#***************************************************
# 2-D Heat Map Data Generation
#**************************************************
import sys
sys.path.insert(0, '../../Utilities/')

import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


plate_length = 10
max_iter_time =10

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Initialize solution: the grid of u(k, i, j)
u = np.empty((max_iter_time, plate_length, plate_length))

# Initial condition everywhere inside the grid
u_initial = 0

# Boundary conditions
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# Set the initial condition
u.fill(u_initial)

# Set the boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length-1):] = u_right

def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]

    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt


# Do the calculation here
u = calculate(u)

output=u.flatten()#This is the solution of 2-D heat equation

#2-D Heat Equation DInput Data Generation X1, X2 and T
q1=np.empty(100) 
q2=np.empty(1000)
q3=np.empty(1000)

q1=np.empty(100)
for i in range(100):
    q1[i]=math.floor(i/10)+1
 
q12= np.tile(q1, (10, 1))
q13=q12.flatten()#X1

for i in range(1000):
    q2[i]=i % 10+1 #X2
    
    

for i in range(1000):
    q3[i]=math.floor(i/100) #T
    
    input = np.stack((q13, q2,q3), axis=1)#Merging X1, X2 and T as three columns
    
N_f=300
sa = np.linspace(100,799, 700)
idx_sample = np.random.choice(sa, N_f, replace=False)#t=0 values are preseved in the original dataset as they are intial conditions

#New train Data for2-D Heat Equation
X_u_train = np.vstack((input[0:100,:],input[idx_sample.astype(int),:]))
u_train = np.concatenate([output[0:100],output[idx_sample.astype(int)]])

#Please use above "input" and "output" data files as testing data.
