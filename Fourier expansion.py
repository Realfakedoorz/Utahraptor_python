# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 00:16:11 2019

@author: Sam
"""

#from DinoBullet import dino3D
import numpy as np
import matplotlib.pyplot as plt

Dino = dino3D()

Dino.popsize = 3

Dino.generateWalk()

M = np.zeros([3, 5, 4])
M = np.array(Dino.population[0])

T_step = 0.01
Time = 20

a0     = M[0]
a1     = M[1]
phase1 = M[2]
a2     = M[3]
phase2 = M[4]
a3     = M[1]
phase3 = M[2]
a4     = M[3]
phase4 = M[4]

w = 0.5

theta = [a0[0]]


for t in range(0,int(Time/T_step),Time):
    theta.append(a0[0] + a1[0]*np.sin(np.deg2rad((0 + phase1[0])/T_step)) 
                +a2[0]*np.sin(np.deg2rad((1*t + phase2[0])/T_step)) 
                +a3[0]*np.sin(np.deg2rad((2*t + phase3[0])/T_step))
                +a4[0]*np.sin(np.deg2rad((3*t + phase4[0])/T_step)))
    
    if t > 0:
        w = (theta[-1]-theta[-2])/T_step

    
    
    