# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 08:40:01 2019

@author: Sam
"""

# -*- coding: utf-8 -*-
"""
Utahraptor Kinematics/Dynamics

Created on Fri Jun 14 08:10:02 2019

@author: Sam
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import matlib as ml
from sympy import *

class leg():
    def __init__(self):
        ''' Initialise parameters '''
        scale = 1
           
        self.m_body = 382*scale;
        self.m_inv = 1/self.m_body
        self.m = [0.,1.,1.,1.,1.]
        
        self.m[1:] = self.muscMasses() # Set masses of parts from Hutchinson's %s
        
        self.L_femur = 565*scale
        self.L_tibia = 505*scale
        self.L_tarsa = 300*scale
        self.L_foot  = 380*scale
        self.w_hips = 200*scale
        
        self.base = np.array([[0, 0], [0, 0], [0, 0]])
        
        self.CoM = [50, 0]
        
        self.resetAngles()
        
        self.forces = np.matrix(ml.repmat(np.zeros([3,1]), 1, 5))
        self.torques = np.matrix(ml.repmat(np.zeros([3,1]), 1, 5))
        
        self.toe_oldx = 0
        
        self.base_old = self.base
        
        ''' DH params '''
        alpha = np.array([0, 0, 0, 0, 0])
        a = np.array([0, self.L_femur, self.L_tibia, self.L_tarsa, self.L_foot])
        d = np.array([0,0,0,0,0])
        
        self.dh = np.array([alpha, a, d])
        
        ''' GA params '''
        self.constraintPenalty = 0
        
        ''' Hold last angle for numerical differentiation '''
        ''' First is last held angle, second is ang velocity '''
        self.w1 = [0, 0]; self.w1 = [0, 0]; self.w3 = [0, 0]; self.w4 = [0, 0]
        
        self.dyn() # Keep me last!
        
    def setFourierParams(self):
        ''' I'm a placeholder! '''
        return
        
    def resetAngles(self):
        ''' Reset angle params '''
        self.th_fem = np.array([0., 0., 0.])
        self.th_kne = np.array([0., 0., 0.])
        self.th_ank = np.array([0., 0., 0.])
        self.th_toe = ([0., 0., 0.])    
        self.base = np.array([[0, 0], [0, 0], [0, -9.81]])
   
    def jointMoments(self):
        ''' Need to add body mass and location '''
        ''' See Hutchinson 2004 
        Unsure how useful this is tho'''
        M = [0, 0, 0, 0, 0]
        m = self.m
        
        joints = self.fkine()
        points = self.fkine(0.5)
        
        M[0] = 0.001*abs(self.CoM[0])*self.m_body
        
        for i in range(1,5):
            M[i] = 0.001*abs(joints[i-1][0] - (self.base[0,0]+self.CoM[0]))*self.m_body
            for j in range(1,i):
                M[i] = M[i] + 0.001*abs(joints[i-1][0]-points[j-1][0])*m[i-1]
        return M
    
    def body_force(self):
        ''' Defines the spring/damping force model of the legs on the body '''
        k = 1.*10**7 # Spring constant 
        y = 2.*10**5 # Damping
        
        #return -(k*(1370-self.euclidean(self.base[0], self.fkine()[2])/1000) + y*self.base[1, 1])
        return self.forces[1,0]*self.m_inv
        
    def angles(self, angles = [0,0,0,0,0]):
        ''' Set or display angles '''
        if angles == [0,0,0,0,0]:
            return [self.th_fem[0], self.th_kne[0], self.th_ank[0], self.th_toe[0]]
        else:
            [self.th_fem[0], self.th_kne[0], self.th_ank[0], self.th_toe[0]] = angles
            return [self.th_fem[0], self.th_kne[0], self.th_ank[0], self.th_toe[0]]
                    
        self.th_fem[0] = -38
        self.th_kne[0] = 180+83
        self.th_ank[0] = 180-110
        self.th_toe[0] = self.foot_angle()
    
    def penConst(self, reset = 0):
        ''' Penalise going against angle constraints '''
        if reset == 0:
            if ((self.th_fem[0]%360 < -75%360) or (self.th_fem[0]%360 > -10%360)):
               self.constraintPenalty += 5
            if ((self.th_kne[0]%360 < (180+83-70)%360) or (self.th_kne[0] > (180+83+50)%360)):
               self.constraintPenalty += 5                
            if ((self.th_ank[0]%360 < (180-110-40)%360) or (self.th_ank[0] > (180-110+40)%360)):
                self.constraintPenalty += 1 
            if self.base[0] < -1750:
                self.constraintPenalty += 20
        else: 
           self.constraintPenalty = 0 
            
    def evolve(self, const=1):
        '''Evolve velocity and distance '''
        self.base[1] += self.base[2]; self.base[0] += self.base[1]
        self.base[2,1] += self.body_force()*self.m_inv
        self.th_fem[1] += self.th_fem[2]; self.th_fem[0] += self.th_fem[1]
        self.th_kne[1] += self.th_kne[2]; self.th_kne[0] += self.th_kne[1]
        self.th_ank[1] += self.th_ank[2]; self.th_ank[0] += self.th_ank[1]
        self.th_toe[1] += self.th_toe[2]; self.th_toe[0] = self.foot_angle()#+= self.th_toe[1]
        
        if const== 1:
            ''' If statements sort angle constraints - set vel to zero on const. '''
            if ((self.th_fem[0]%360 < -75%360) or (self.th_fem[0]%360 > -25%360)):
                self.th_fem[0] -= self.th_fem[1]
                self.th_fem[1] = 0#needs mod720 - 360??
            
            if ((self.th_kne[0]%360 < (180-60)%360) or (self.th_kne[0] > (180+60)%360)):
                self.th_kne[0] -= self.th_kne[1]
                self.th_kne[1] = 0
            
            if ((self.th_ank[0]%360 < (180-160)%360) or (self.th_ank[0] > (180-60)%360)):
                self.th_ank[0] -= self.th_ank[1]
                self.th_ank[1] = 0
        
    def fkine(self, pos="Full"):    
        ''' Forward kinematics to produce end co-ords of each joint '''
        th = np.array([0., 0., self.th_fem[0], self.th_kne[0], self.th_ank[0], self.th_toe[0], 0.], dtype = float)
        #old = np.array([0., 0., -self.th_fem[0], 180+self.th_kne[0], 180-self.th_ank[0], self.th_toe[0], 0.]
        
        
        for i in range(0,th.size):
            th[i] = th[i] *np.pi/180
        
        c2 = np.cos(th[2])
        s2 = np.sin(th[2])
        c4 = np.cos(th[4])
        s4 = np.sin(th[4]);
        c23 = np.cos(th[2] + th[3])
        s23 = np.sin(th[2] + th[3])
        c45 = np.cos(th[4] + th[5]) 
        s45 = np.sin(th[4] + th[5])
        
        femur = self.base[0]+np.array([self.L_femur*c2, self.L_femur*s2])
        tibia = femur+  np.array([self.L_tibia*c23, self.L_tibia*s23])
        tarsus = tibia+ np.array([self.L_tarsa*(c4*c23 - s4*s23), self.L_tarsa*(c4*s23 + s4*c23)])
        toe = tarsus+   np.array([self.L_foot*(c23*c45 - s23*s45), self.L_foot*(c23*s45 + s23*c45)])
        
        
        ''' Ground constraint '''
        if toe[1] <= -1750:
            self.base[0,0] -= toe[0]-self.toe_oldx; self.base[0,1] += abs(toe[1] + 1750)
            femur[0] -= toe[0]-self.toe_oldx; femur[1] += abs(toe[1] + 1750)
            tibia[0] -= toe[0]-self.toe_oldx; tibia[1] += abs(toe[1] + 1750)
            tarsus[0] -= toe[0]-self.toe_oldx; tarsus[1] += abs(toe[1] + 1750)
            toe[0] = self.toe_oldx; toe[1] = -1750
            #print("Constraint hit")
            
    
        self.toe_oldx = toe[0]
        
        self.base_old = self.base
        
        if tarsus[1] < toe[1]:
            self.constraintPenalty += 20
        if tibia[1] < toe[1]:
            self.constraintPenalty += 30
        if femur[1] < toe[1]:
            self.constraintPenalty += 40
        
        if pos!="Full":
            a = self.base[0]+np.array([0.5*self.L_femur*c2, 0.5*self.L_femur*s2])
            b = femur+  np.array([0.5*self.L_tibia*c23, 0.5*self.L_tibia*s23])
            c = tibia+ np.array([0.5*self.L_tarsa*(c4*c23 - s4*s23), 0.5*self.L_tarsa*(c4*s23 + s4*c23)])
            d = tarsus+   np.array([0.5*self.L_foot*(c23*c45 - s23*s45), 0.5*self.L_foot*(c23*s45 + s23*c45)])
            return np.array([a, b, c, d])
        else:
            return np.array([femur, tibia, tarsus, toe])
    
    def transform(self, frm, to):
        ''' Find frame transformations '''
        T = np.eye(4)
        
        ''' Unpack DH params '''
        alpha = self.dh[0]
        a = self.dh[1]
        d = self.dh[2]
        th = np.array([self.th_fem[0], self.th_kne[0], self.th_ank[0], self.th_toe[0], 0], dtype=float)
        
        for i in range(0,th.size):
            th[i] = th[i] *np.pi/180
        
        for i in range(frm,to):
            ct = np.cos(th[i])
            st = np.sin(th[i])
            ca = np.cos(alpha[i])
            sa = np.sin(alpha[i])
            
            if abs(ct) < 1e-15:
                ct = 0
            if abs(st) < 1e-15:
                st = 0
            if abs(ca) < 1e-15:
                ca = 0
            if abs(sa) < 1e-15:
                sa = 0
            
            T = np.dot(T, np.array([[ct, -st*ca,  st*sa, a[i]*ct],
                          [st,  ct*ca, -ct*sa, a[i]*st],
                          [0,   sa,     ca,    d[i]   ],
                          [0,   0,      0,     1]])) 
        return T

    def dyn(self, pretty_output=0):
        ''' Inverse dynamics using recursive Newton-Euler. Requires verification
        Seems to be sorted!
        '''
        N_dof = 5
        
        '''Unpack DH params'''
        alpha = self.dh[0]
        a = np.array(self.dh[1])/1000
        d = self.dh[2]
        #th = np.array([0., -self.th_fem[0], 180.+self.th_kne[0], 180.-self.th_ank[0], self.th_toe[0]], dtype=float)
           
        ''' Link RB params '''
        m = np.matrix(self.m) # Link masses
        #I = np.array([np.eye(3), np.eye(3), np.eye(3),np.eye(3),np.eye(3)])   # Inertia matrix
        #r = np.transpose(np.matrix([[-0.5*self.w_hips, 0, 0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]], dtype=float)) # CoMs from end effector
        # Assuming point mass to avoid inertia
        
        ''' CoMs '''
        p_c = np.transpose(np.matrix([[0, 0, 0], [self.L_femur/2, 0, 0], [self.L_tibia/2, 0, 0], [self.L_tarsa/2, 0, 0], [self.L_foot/2, 0, 0]]))/1000
        
        ''' Define ground reaction force '''
        ''' Use an if to include only on touching the ground '''
        F_ground = np.matrix([[0], [2.5*self.m_body], [0]], dtype = float)
        
        ''' Add gravity force '''
        g = np.matrix([[0], [-9.81], [0]], dtype=float)
        
        ''' Add direction of first joint ie. centre of hip to femur '''
        z0 = np.matrix([[0], [0], [1]], dtype=float)
        
        qc = np.matrix([[0, 0, 0], self.th_fem, self.th_kne, self.th_ank, self.th_toe])
        
        w = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        wdot = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        vdot = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        vcdot = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        n = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        f = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        F = np.matrix(ml.repmat(np.zeros([3,1]), 1, N_dof))
        p = np.matrix(np.zeros([1,3])) ### Some problem with p?     
        
        # Forwards
        for i in range(1, N_dof+1):#
            T = self.transform(i-1, i)#
            R = np.matrix(T[0:3, 0:3]);#
            
            p = np.matrix([[a[i-1]], [d[i-1]*np.sin(alpha[i-1])], [d[i-1]*np.cos(alpha[i-1])]], dtype = float)#
            
            if i > 1:#
                w[:, i-1]       = np.transpose(R)@(w[:, i-2] + np.dot(z0, qc[i-1,1]))
                wdot[:, i-1]    = np.transpose(R)@(wdot[:, i-2] +  np.dot(z0, qc[i-1,2]) + np.cross(w[:, i-2], np.dot(z0, qc[i-1,1]), axis=0))#
                vdot[:, i-1]    = np.transpose(R)@vdot[:, i-2] + np.cross(wdot[:, i-1], p, axis=0) + np.cross(w[:, i-1], np.cross(w[:, i-1],p, axis=0), axis=0)#
                vcdot[:, i-1]   = np.cross(w[:,i-1], p_c[:,i-1], axis=0) + np.cross(w[:, i-1], np.cross(w[:, i-1], p_c[:, i-1], axis=0), axis=0) + vdot[:, i-1]#
                F[:, i-1]       = np.asscalar(m[:,i-1])*vcdot[:,i-1]#
            else:
                w[:, i-1]       = np.transpose(R)@np.dot(z0, qc[i-1, 1])#
                wdot[:, i-1]    = np.transpose(R)@np.dot(z0, qc[i-1, 2])#
                vdot[:, i-1]    = np.transpose(R)@g + np.cross(wdot[:, i-1], p, axis=0) + np.cross(w[:, i-1], np.cross(w[:, i-1],p, axis=0), axis=0)#
                vcdot[:, i-1] = np.cross(w[:, i-1], p_c[:, i-1], axis=0) + np.cross(w[:, i-1], np.cross(w[:, i-1], p_c[:, i-1], axis=0), axis=0) + vdot[:, i-1]#
                F[:, i-1]       = np.asscalar(m[:,i-1])*vcdot[:, i-1]#

        # Backwards
        for i in range(N_dof, 0, -1):#
            p = np.matrix([[a[i-1]], [d[i-1]*np.sin(alpha[i-1])], [d[i-1]*np.cos(alpha[i-1])]], dtype = float)#
            if i < N_dof:#??
                T = self.transform(i-1, i)#
                R = T[0:3, 0:3]#
                
                n[:, i-1] = R@(n[:, i] + np.cross(p,f[:, i], axis=0)) + np.cross(p_c[:, i-1],F[:, i-1], axis=0)#
                f[:, i-1] = R@f[:, i] + F[:, i-1]#
            else:
                T = self.transform(0, 5)#
                R = T[0:3, 0:3]#
                
                n[:, i-1] = np.cross(p_c[:, i-1],F[:, i-1], axis=0)#
                if self.fkine()[3,1]<=-1750:
                    f[:, i-1] = F[:, i-1] + np.transpose(R)*F_ground#
                else:
                    f[:, i-1] = F[:, i-1]
            
            T = self.transform(i-1, i)#
            R = T[1:3, 1:3]#      

        self.torques = n
        self.forces = f
        
        if pretty_output != 0:
            print("Forces in frames:")
            print(f)
            print("Torques about frame axes")
            print(n)
        
        return n[2, :]
    
    def muscMasses(self):
        ''' Calculate muscle masses. Well out '''
        th = np.array([15, 50, 90, 120])*np.pi/180
        mi = np.dot(th, 0)
        '''
        G = 3 # Activity level
        g = 9.81
        d = 1.06*10**3
        sigma = 3.00 * 10**5
        c = 1.
        R = [0.06, 0.04, 0.09, 0.03]
        r = [0.1, 0.04, 0.04, 0.016]
        L = [self.L_femur/4000, 0.04, 0.04, 0.04]
        
        
        for i in range(0,4):
            #mi[i] = (100*G*g*R[i]*L[i]*d)/(sigma*c*r[i]*np.cos(th[i]))
            mi[i] = 1.767*R[i]*L[i]/r[i]
        '''
        # Using averages here 
        mi = np.array([0.07, 0.05, 0.01, 0.006])*self.m_body
        
        return mi
    
    def foot_angle(self):
        ''' Calculate angle to keep foot flat '''
        return -np.sum([self.th_ank[0], self.th_fem[0], self.th_kne[0]])
    
    def show(self):
        plt.close("all")
        
        ''' Initialise plot '''
        fig = plt.figure()
        ax = plt.axes(xlim=(-2000, 2000), ylim=(-2000,2000))
        line1, = ax.plot([], [], 'k-')
        line2, = ax.plot([], [], 'k-')
        line3, = ax.plot([], [], 'k-')
        line4, = ax.plot([], [], 'k-')
        joints, = ax.plot([], [], 'rs')
        coms,   = ax.plot([], [], 'bx')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        
        [a,b,c,d] = self.fkine()
        [e,f,g,h] = self.fkine(0.5)
            
        line1.set_data([self.base[0, 0], a[0]], [self.base[0, 1], a[1]])
        line2.set_data([a[0], b[0]], [a[1], b[1]])
        line3.set_data([b[0], c[0]], [b[1], c[1]])
        line4.set_data([c[0], d[0]], [c[1], d[1]])
        joints.set_data([self.base[0, 0], a[0], b[0], c[0], d[0]], [self.base[0, 1], a[1], b[1], c[1], d[1]])
        coms.set_data([e[0], f[0], g[0], h[0]], [e[1], f[1], g[1], h[1]])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
            
    def test(self, test=1):
        ''' Deprecate, it's working '''
        if test == 1:
            self.resetAngles()
        elif test == 2:
            self.base[0] = [0,0]
            self.angles([-90,0,0,0])
        else:
            self.base[0] = [0,0]
            self.th_fem[0] = -38
            self.th_kne[0] = 180+83
            self.th_ank[0] = 180-110
            self.th_toe[0] = self.foot_angle()
        
    def euclidean(self, a, b):
        ''' Helper, calculates point to point distance '''
        return np.power(np.power(a[0]-b[0], 2) + np.power(a[1]-b[1], 2), 0.5)
    
    
        
        