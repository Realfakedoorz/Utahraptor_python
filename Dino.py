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
from leg import *
from numpy import random as rd
import pickle
#import pybullet as pb
#import pybullet_data
import DinoPb as pb
import matplotlib.animation as manimation

class dino():
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
        
        self.base = np.array([[0, 0], [0, 0], [0, -9.81]])
        
        self.CoM = [50, 0]
        
        self.resetAngles()
        
        self.forces = np.matrix(ml.repmat(np.zeros([3,1]), 1, 5))
        self.torques = np.matrix(ml.repmat(np.zeros([3,1]), 1, 5))
        
        self.toe_oldx = 0
        
        ''' DH params '''
        alpha = np.array([0, 0, 0, 0, 0])
        a = np.array([0, self.L_femur, self.L_tibia, self.L_tarsa, self.L_foot])
        d = np.array([0,0,0,0,0])
        
        self.dh = np.array([alpha, a, d])
        
        ''' GA params '''
        #self.amp = 
        self.t0s = [0,0,0,0]
        self.tas = [0,0,0,0]
        self.tTs = [0,0,0,0]
        
        self.popsize = 40
        self.population = np.zeros([self.popsize,3,4])
        self.parents    = np.zeros([2,3,4])
        self.seed = rd.randint(0,10000)
        self.fitnesses = np.zeros([self.popsize,1])
        self.fittotal = 0
        self.fitinhand = -99999
        self.inhand = 0
        self.generate_population()
        self.SL = 0
        
        ''' pyBullet params '''
        self.T_fixed = 1./240

        
        ''' Hold last angle for numerical differentiation '''
        ''' First is last held angle, second is ang velocity '''
        self.w1 = [0, 0]; self.w1 = [0, 0]; self.w3 = [0, 0]; self.w4 = [0, 0]
        
        self.dyn() # Keep me last!
        
        self.leg_l = leg(); self.leg_l.test(3)
        self.leg_r = leg(); self.leg_r.test(3)


    def pbConnect(self):
        physicsClient = pb.connect(pb.GUI)
        if physicsClient < 0:
            pb.disconnect()
            physicsClient = pb.connect(pb.GUI)
        
        pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        pb.setGravity(0,0,-9.81)
        pb.setPhysicsEngineParameter(fixedTimeStep = self.T_fixed, 
                                     solverResidualThreshold = 1 - 10, 
                                     numSolverIterations = 50, 
                                     numSubSteps = 4)
        
        planeId = pb.loadURDF("plane100.urdf")
        
        self.botStartPos = [0,0,2]
        self.botStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.botID = pb.loadURDF("./Model/utahraptor.urdf",self.botStartPos, self.botStartOrientation)
        
        #self.botId = pb.loadURDF("R2D2.urdf",self.botStartPos, self.botStartOrientation)

        self.numJoints = pb.getNumJoints(self.botID)
        for i in range(0,self.numJoints):
            pb.setJointMotorControl2(bodyUniqueId = self.botID, jointIndex = i, controlMode = pb.POSITION_CONTROL, force = 0)
        
    def pbStep(self, steps = 10000):
        for i in range(0, steps):
            pb.stepSimulation()
            botPos, botOrn = pb.getBasePositionAndOrientation(self.botID)
            print("\n")
            print(botPos,botOrn)
            
            pb.resetDebugVisualizerCamera( cameraDistance=5, 
                                          cameraYaw=0.0, 
                                          cameraPitch=-20.0, 
                                          cameraTargetPosition=botPos)
            time.sleep(self.T_fixed)
            
        
    def pbDisconnect(self):
        pb.disconnect()


    ''' 2D GA related functions '''

    def GA(self, epochs = 3):
        ''' Run genetic algorithm '''
        change = 0
        mutationrate = 0.2
    
        for ep in range(0,epochs):
            self.fittotal = 0
            '''
            self.generate_population()
            if ep > 0:
                self.population[0:2] = self.parents
            '''
            best = [-99999999, -99999999]
            bestid = [0, 0]     
            for i in range(0,self.popsize):
                if ep > 0:
                    self.population[0:2] = self.parents
            
                self.fitnesses[i,0] = self.fitness(i); self.fittotal += self.fitnesses[i,0]
                
                if self.fitnesses[i,0] > max(best):
                    bestid[0] = i
                    best[0] = self.fitnesses[i,0]
                    change += 1
                elif self.fitnesses[i,0] > min(best) and self.fitnesses[i,0] != max(best):
                    bestid[1] = i
                    best[1] = self.fitnesses[i,0]
                    change += 1
            
            self.parents[0] = self.population[bestid[0]]
            self.parents[1] = self.population[bestid[1]]
            if max(best) > self.fitinhand:
                self.fitinhand = max(best)
                self.inhand = self.parents[0]
                print("Fitness: {}".format(self.fitinhand))
            #self.population[0:2] = self.parents
            self.selection()
            self.blend()
            
            if mutationrate > 0.01:
                mutationrate = 0.2*(1-ep/(0.7*epochs))
            else:
                mutationrate = 0.01
                
            self.mutate(mutationrate)
            print(".")    
            if ep%2==0:
                print("{:.2f}% complete".format(100*ep/epochs))
                print("Rate of mutation: {:.0f}%".format(100*mutationrate))
        print('\007')
        self.animate_GA(100)
        
    def generate_population(self):
        ''' Generate 20 random sets of parameters '''
        rd.seed()
        for d in range(0,self.popsize):
            for w in range(0,4):
                self.population[d, 0, w] = rd.randint(0,360) #theta0xs
                self.population[d, 1, w] = rd.randint(-30,30)#amplitudes
                self.population[d, 2, w] = rd.randint(0,50)#Ts
       
    def selection(self, select_n = 10):
        ''' Roulette wheel and crossover '''
        pop_tmp = self.population
        
        self.fittotal = 0
        
        for i in range(0,self.popsize):
            self.fitnesses[i,0] = self.fitness(i); self.fittotal += self.fitnesses[i,0]
        
        probabilities = np.round(self.fitnesses/self.fittotal, decimals = 3)
        # Fudge to make sum(p) = 1
        if np.sum(probabilities != 1):            
            n = 0
            s = np.sum(probabilities)
            while probabilities[n] < s-1:
                n += 1
                
            probabilities[n] += 1-np.sum(probabilities)
        
        selections = np.ones([select_n])*999
        
        
        for i in range(0,select_n):
            choose = rd.choice(range(0, len(self.population)), p = np.ndarray.tolist(probabilities[:,0]))
            limit = 0
            while any(selections == float(choose)):
                limit += 1
                choose = rd.choice(range(0, len(self.population)), p = np.ndarray.tolist(probabilities[:,0]))
                if limit == 20:
                    print("break")
                    limit = 0
                    break
            selections[i] = int(choose)

        #self.generate_population()
        for i in range(0, select_n):
            self.population[i] = pop_tmp[int(selections[i])]
    
        for i in range(0,self.popsize):
            self.fitnesses[i,0] = self.fitness(i); self.fittotal += self.fitnesses[i,0]
    
    def crossover(self, select_n = 10):
        ''' Blend (BLX-alpha) best parents up to generate the children '''
        sort = np.argsort(-self.fitnesses[:,0])
        parent1 = self.population[sort[0]]
        parent2 = self.population[sort[1]]
        for i in range(0, self.popsize):
            for j in range(0, len(self.population[0,:])):
                for k in range(0, len(self.population[0,0,:])):
                    if rd.random() < 0.5:
                        self.population[i, j, k] = parent1[j,k]
                    else:
                        self.population[i, j, k] = parent2[j,k]
    
    def blend(self, select_n = 10):
        ''' Blend (BLX-alpha) best parents up to generate the children '''
        sort = np.argsort(-self.fitnesses[:,0])
        
        n = select_n
        
        while n < self.popsize:
            parent1 = self.population[sort[(0+n)%select_n]]
            parent2 = self.population[sort[(1+n)%select_n]]
            
            for j in range(0, len(self.population[0,:])):
                for k in range(0, len(self.population[0,0,:])):
                    d = abs(parent1[j,k]-parent2[j,k])
                    
                    self.population[n,j,k] = min([parent1[j,k],parent2[j,k]])+rd.random()*d
            n+=1
            
    def mutate(self, rate = 0.2):
        ''' Randomly modify parameters '''
        for i in range(0,self.popsize):
            for w in range(0, len(self.population[0,0,:])):
                if rd.rand() < rate:
                    self.population[i, 0, w] = rd.randint(0,360) #theta0xs
                if rd.rand() < rate:
                    self.population[i, 1, w] = rd.randint(-30,30)#amplitudes
                if rd.rand() < rate:
                    self.population[i, 2, w] = rd.randint(0,50)#Ts
            
    def fitness(self, num):
        ''' Test fitness of a member of the population '''
        T = 1
        T_step = 0.1
        legt0 = self.population[num, 0]
        legamp = self.population[num, 1]
        legT = self.population[num, 2]
        
        
        self.resetAngles()
        legs = [self.leg_l, self.leg_r]
        for l in legs:
            l.resetAngles()
            l.penConst(1)
    
        for i in range(0,10):
            t = float(i)*(T_step)
            self.leg_l.th_fem[0] = legt0[0] + legamp[0]*np.sin(2*np.pi*(t + legT[0])/T)
            self.leg_l.th_kne[0] = legt0[1] + legamp[1]*np.sin(2*np.pi*(t + legT[1])/T)
            self.leg_l.th_ank[0] = legt0[2] + legamp[2]*np.sin(2*np.pi*(t + legT[2])/T)
            self.leg_l.th_toe[0] = self.leg_l.foot_angle()
            
            self.leg_r.th_fem[0] = legt0[0] + legamp[0]*np.sin(np.pi + 2*np.pi*(t + legT[0])/T)
            self.leg_r.th_kne[0] = legt0[1] + legamp[1]*np.sin(np.pi + 2*np.pi*(t + legT[1])/T)
            self.leg_r.th_ank[0] = legt0[2] + legamp[2]*np.sin(np.pi + 2*np.pi*(t + legT[2])/T)
            self.leg_r.th_toe[0] = self.leg_r.foot_angle()
            
            for l in legs:
                l.dyn()
            
            
            for legidx in range(0,2):
                [a, b, c, d] = legs[legidx].fkine()
                off = self.matchLegs()
                a += off 
                b += off
                c += off
                d += off
            
            self.evolve()
            self.matchLegs()            
            self.leg_l.evolve(0)
            self.leg_r.evolve(0)
            self.matchLegs()
        
        distpos = self.base[0,0]
        if distpos < 0:
            distpos = 0
            
        height = self.base[0,1]
        
        ''' Keep CoM within legs '''
        balance = 0
        if (self.base[0,0] < min([self.leg_l.toe_oldx, self.leg_r.toe_oldx])) and (self.base[0,0] > max([self.leg_l.toe_oldx, self.leg_r.toe_oldx])):
            balance = -400000
        
                    
        ft = 100000 + 700*distpos - 3*(self.leg_l.constraintPenalty+self.leg_r.constraintPenalty) + 9*(height-450) + balance
        if ft > 0:
            return (ft)
        else:
            return 0
    
    
    ''' Graphics related functions '''
    def animate_GA(self, steps=10, num = -99):
        ''' Create an animated plot of leg motion '''
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='', artist='',
                        comment='')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        
        vid = []
        fig = plt.figure()
        img = [[1, 2, 3] * 50] * 100
        l = plt.imshow(img, interpolation='none', animated=True, label="blah")
        self.vid.append(self.vid[-1])
        with writer.saving(fig, name, 100):
            for i in range(len(self.vid)):
                l.set_data(self.vid[i])
                writer.grab_frame()
        self.vid = []
        
        
        T = 1
        T_step = 0.1
        if num == -99:
            legt0 = self.inhand[0]
            legamp = self.inhand[1]
            legT = self.inhand[2]
        else:
            legt0 = self.population[num,0]
            legamp = self.population[num, 1]
            legT = self.population[num,2]
        
        plt.close("all")
        
        fig = plt.figure()
        ax = plt.axes(xlim=(-2000, 2000), ylim=(-2000,2000))
        line1 = [0,0]
        line2 = [0,0]
        line3 = [0,0]
        line4 = [0,0]
        joints = [0,0]
        coms = [0,0]
        
        line1[0], = ax.plot([], [], 'g-'); line1[1], = ax.plot([], [], 'm-')
        line2[0], = ax.plot([], [], 'g-'); line2[1], = ax.plot([], [], 'm-')
        line3[0], = ax.plot([], [], 'g-'); line3[1], = ax.plot([], [], 'm-')
        line4[0], = ax.plot([], [], 'g-'); line4[1], = ax.plot([], [], 'm-')
        joints[0], = ax.plot([], [], 'rs'); joints[1], = ax.plot([], [], 'rs')
        coms[0],   = ax.plot([], [], 'bx'); coms[1], = ax.plot([], [], 'bx')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        
        self.resetAngles()
        self.leg_l.resetAngles()
        self.leg_r.resetAngles()
        legs = [self.leg_l, self.leg_r]
        self.torquelist = []
        for i in range(0,steps):
            ax.set_title("Step = {}".format(i))
            
            t = float(i)*(T_step)
            self.leg_l.th_fem[0] = legt0[0] + legamp[0]*np.sin(2*np.pi*(t + legT[0])/T); #print(self.leg_l.th_fem[0])
            self.leg_l.th_kne[0] = legt0[1] + legamp[1]*np.sin(2*np.pi*(t + legT[1])/T)
            self.leg_l.th_ank[0] = legt0[2] + legamp[2]*np.sin(2*np.pi*(t + legT[2])/T)
            self.leg_l.th_toe[0] = self.leg_l.foot_angle()#legt0[3] + legamp[3]*np.sin(2*np.pi*(t + legT[3])/T)
            
            self.leg_r.th_fem[0] = legt0[0] + legamp[0]*np.sin(np.pi + 2*np.pi*(t + legT[0])/T)
            self.leg_r.th_kne[0] = legt0[1] + legamp[1]*np.sin(np.pi + 2*np.pi*(t + legT[1])/T)
            self.leg_r.th_ank[0] = legt0[2] + legamp[2]*np.sin(np.pi + 2*np.pi*(t + legT[2])/T)
            self.leg_r.th_toe[0] = self.leg_r.foot_angle() #legt0[3] + legamp[3]*np.sin(np.pi + 2*np.pi*(t + legT[3])/T)
            
            for l in legs:
                l.dyn()
            self.torquelist.append(self.leg_l.torques[2, :])
            
            
            
            for legidx in range(0,2):
                [a, b, c, d] = legs[legidx].fkine()
                off = self.matchLegs()
                a += off 
                b += off
                c += off
                d += off
                
                ''' Legs are separating due to ground constraint conflicts I believe '''
                line1[legidx].set_data([self.base[0, 0], a[0]], [self.base[0, 1], a[1]])
                line2[legidx].set_data([a[0], b[0]], [a[1], b[1]])
                line3[legidx].set_data([b[0], c[0]], [b[1], c[1]])
                line4[legidx].set_data([c[0], d[0]], [c[1], d[1]])
                joints[legidx].set_data([legs[legidx].base[0, 0], a[0], b[0], c[0], d[0]], [legs[legidx].base[0, 1], a[1], b[1], c[1], d[1]])
                ax.set_xlim(self.base[0,0]-2000, self.base[0,0]+2000)
            
            self.evolve()
            self.matchLegs()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            ''' Allow early break on window close '''
            if plt.fignum_exists(fig.number) == 0:
                #self.resetAngles()
                break
            
            ''' Slide xaxis and change text '''
            time.sleep(0.1)
            
            self.leg_l.evolve(0)
            self.leg_r.evolve(0)
            self.matchLegs()


    def animate_both(self, steps=40):
        ''' Create an animated plot of leg motion '''
        T = 0.1 # Time of 1 step
        plt.close("all")
        
        fig = plt.figure()
        ax = plt.axes(xlim=(-2000, 2000), ylim=(-2000,2000))
        line1 = [0,0]
        line2 = [0,0]
        line3 = [0,0]
        line4 = [0,0]
        joints = [0,0]
        coms = [0,0]
        
        line1[0], = ax.plot([], [], 'g-'); line1[1], = ax.plot([], [], 'm-')
        line2[0], = ax.plot([], [], 'g-'); line2[1], = ax.plot([], [], 'm-')
        line3[0], = ax.plot([], [], 'g-'); line3[1], = ax.plot([], [], 'm-')
        line4[0], = ax.plot([], [], 'g-'); line4[1], = ax.plot([], [], 'm-')
        joints[0], = ax.plot([], [], 'rs'); joints[1], = ax.plot([], [], 'rs')
        coms[0],   = ax.plot([], [], 'bx'); coms[1], = ax.plot([], [], 'bx')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        
        legs = [self.leg_l, self.leg_r]
    
        for i in range(0,steps):
            ax.set_title("Step = {}".format(i))
            t = float(i)*np.pi/180
            self.leg_l.th_fem[2] = 1*np.cos(5*t)
            self.leg_l.th_kne[2] = -1*np.sin(7*t)
            self.leg_l.th_ank[2] = 2*np.cos(3*t)
            self.leg_l.th_toe[2] = 1*np.cos(11*t)
            
            self.leg_r.th_fem[2] = 4*np.cos(6*t)
            self.leg_r.th_kne[2] = -7*np.sin(3*t)
            self.leg_r.th_ank[2] = 8*np.cos(t)
            self.leg_r.th_toe[2] = 11*np.cos(2*t)
            
            for l in legs:
                l.dyn()
            
            
            for legidx in range(0,2):
                [a, b, c, d] = legs[legidx].fkine()
                off = self.matchLegs()
                a += off 
                b += off
                c += off
                d += off
                
                ''' Legs are separating due to ground constraint conflicts I believe '''
                line1[legidx].set_data([self.base[0, 0], a[0]], [self.base[0, 1], a[1]])
                line2[legidx].set_data([a[0], b[0]], [a[1], b[1]])
                line3[legidx].set_data([b[0], c[0]], [b[1], c[1]])
                line4[legidx].set_data([c[0], d[0]], [c[1], d[1]])
                joints[legidx].set_data([legs[legidx].base[0, 0], a[0], b[0], c[0], d[0]], [legs[legidx].base[0, 1], a[1], b[1], c[1], d[1]])
                ax.set_xlim(self.base[0,0]-2000, self.base[0,0]+2000)
            
            self.evolve()
            self.matchLegs()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            ''' Allow early break on window close '''
            if plt.fignum_exists(fig.number) == 0:
                #self.resetAngles()
                break
            
            ''' Slide xaxis and change text '''
            time.sleep(T)
            
            self.leg_l.evolve(0)
            self.leg_r.evolve(0)
            self.matchLegs()

    def showboth(self):
        plt.close("all")
        
        ''' Initialise plot '''
        fig = plt.figure()
        ax = plt.axes(xlim=(-2000, 2000), ylim=(-2000,2000))
        line1 = [0,0]
        line2 = [0,0]
        line3 = [0,0]
        line4 = [0,0]
        joints = [0,0]
        coms = [0,0]
        
        line1[0], = ax.plot([], [], 'g-'); line1[1], = ax.plot([], [], 'm-')
        line2[0], = ax.plot([], [], 'g-'); line2[1], = ax.plot([], [], 'm-')
        line3[0], = ax.plot([], [], 'g-'); line3[1], = ax.plot([], [], 'm-')
        line4[0], = ax.plot([], [], 'g-'); line4[1], = ax.plot([], [], 'm-')
        joints[0], = ax.plot([], [], 'rs'); joints[1], = ax.plot([], [], 'rs')
        coms[0],   = ax.plot([], [], 'bx'); coms[1], = ax.plot([], [], 'bx')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        
        legs = [self.leg_l, self.leg_r]
        
        for legix in range(0,2):
            [a,b,c,d] = legs[legix].fkine()
            [e,f,g,h] = legs[legix].fkine(0.5)
                
            line1[legix].set_data([legs[legix].base[0, 0], a[0]], [legs[legix].base[0, 1], a[1]])
            line2[legix].set_data([a[0], b[0]], [a[1], b[1]])
            line3[legix].set_data([b[0], c[0]], [b[1], c[1]])
            line4[legix].set_data([c[0], d[0]], [c[1], d[1]])
            joints[legix].set_data([legs[legix].base[0, 0], a[0], b[0], c[0], d[0]], [legs[legix].base[0, 1], a[1], b[1], c[1], d[1]])
            coms[legix].set_data([e[0], f[0], g[0], h[0]], [e[1], f[1], g[1], h[1]])
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
        
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
    
    def animate_test(self, steps=40):
        ''' Create an animated plot of leg motion '''
        self.test(3)
        T = 0.1 # Time of 1 step
        plt.close("all")
        
        ''' Initialise plot '''
        fig = plt.figure()
        ax = plt.axes(xlim=(-2000, 2000), ylim=(-2000,2000))
        line1, = ax.plot([], [], 'k-')
        line2, = ax.plot([], [], 'k-')
        line3, = ax.plot([], [], 'k-')
        line4, = ax.plot([], [], 'k-')
        joints, = ax.plot([], [], 'rs')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        fig.show()
        tx = ax.text(1200,500, "Torques (Nm):\n {:d},\n {:d},\n {:d},\n {:d},\n {:d}\n".format(int(self.torques[2,0]), int(self.torques[2,1]), int(self.torques[2,2]), int(self.torques[2,3]), int(self.torques[2,4])))
        #tx = ax.text(-10,500, "{}".format(self.body_force()))
        
        ''' Iterate animation '''
        for i in range(0,steps):
            ax.set_title("Step = {}".format(i))
            t = float(i)*np.pi/180
            self.th_fem[2] = 1*np.cos(5*t)
            self.th_kne[2] = -1*np.sin(7*t)
            self.th_ank[2] = 2*np.cos(3*t)
            self.th_toe[2] = 1*np.cos(11*t)
            
            [a,b,c,d] = self.fkine()
            if i%10 == 0 and d[1] <= -1750:
                self.base[0,1] += 500
                self.base[1,1] = 0
                a[1] += 500
                b[1] += 500
                c[1] += 500
                d[1] += 500
                print("Jump")
            
            line1.set_data([self.base[0, 0], a[0]], [self.base[0, 1], a[1]])
            line2.set_data([a[0], b[0]], [a[1], b[1]])
            line3.set_data([b[0], c[0]], [b[1], c[1]])
            line4.set_data([c[0], d[0]], [c[1], d[1]])
            joints.set_data([self.base[0, 0], a[0], b[0], c[0], d[0]], [self.base[0, 1], a[1], b[1], c[1], d[1]])
            fig.canvas.draw()
            fig.canvas.flush_events()
            self.dyn()
            
            ''' Allow early break on window close '''
            if plt.fignum_exists(fig.number) == 0:
                #self.resetAngles()
                break
            
            ''' Slide xaxis and change text '''
            tx.remove()
            ax.set_xlim(self.base[0,0]-2000, self.base[0,0]+2000)
            tx = ax.text(plt.xlim()[1]-1000,500, "Torques (Nm):\n {:d},\n {:d},\n {:d},\n {:d},\n {:d}\n".format(int(self.torques[2,0]), int(self.torques[2,1]), int(self.torques[2,2]), int(self.torques[2,3]), int(self.torques[2,4])))
            
            time.sleep(T)
            
            self.evolve()
            
            
        
        
    ''' General worker functions '''
    def matchLegs(self, reset = 0):
        ''' Links motion of base to both legs '''
        offset = (self.leg_l.base-self.base) + (self.leg_r.base - self.base)
        if reset == 1:
            self.base = np.array([[0, 0], [0, 0], [0, -9.81]])
            self.leg_l.base = np.array([[0, 0], [0, 0], [0, -9.81]])
            self.leg_r.base = np.array([[0, 0], [0, 0], [0, -9.81]])
        else:
            self.base += offset
            self.leg_l.base= self.base
            self.leg_r.base = self.base
        return offset[0]
    
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
        #k = 1.*10**7 # Spring constant 
        #y = 2.*10**5 # Damping
        
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
        
    def evolve(self):
        '''Evolve velocity and distance '''
        self.base[1] += self.base[2]; self.base[0] += self.base[1]
        
        self.th_fem[1] += self.th_fem[2]; self.th_fem[0] += self.th_fem[1]
        ''' If statements sort angle constraints - set vel to zero on const. '''
        if ((self.th_fem[0]%360 < -75%360) or (self.th_fem[0]%360 > -25%360)):
            self.th_fem[0] -= self.th_fem[1]
            self.th_fem[1] = 0#needs mod720 - 360??
        
        self.th_kne[1] += self.th_kne[2]; self.th_kne[0] += self.th_kne[1]
        if ((self.th_kne[0]%360 < (180-60)%360) or (self.th_kne[0] > (180+60)%360)):
            self.th_kne[0] -= self.th_kne[1]
            self.th_kne[1] = 0
        
        self.th_ank[1] += self.th_ank[2]; self.th_ank[0] += self.th_ank[1]
        if ((self.th_ank[0]%360 < (180-160)%360) or (self.th_ank[0] > (180-60)%360)):
            self.th_ank[0] -= self.th_ank[1]
            self.th_ank[1] = 0
            
        self.th_toe[1] += self.th_toe[2]; self.th_toe[0] = self.foot_angle()#+= self.th_toe[1]
        
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
            
    
        self.toe_oldx = toe[0]
        
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
    
    def foot_angle(self):
        ''' Calculate angle to keep foot flat '''
        return -np.sum([self.th_ank[0], self.th_fem[0], self.th_kne[0]])
            
    def test(self, test=1):
        ''' Deprecate, it's working '''
        if test == 1:
            self.resetAngles()
        elif test == 2:
            self.resetAngles()
            self.angles([-90,0,0,0])
        else:
            self.resetAngles()
            self.th_fem[0] = -38
            self.th_kne[0] = 180+83
            self.th_ank[0] = 180-110
            self.th_toe[0] = self.foot_angle()
        
        self.show()
        
    def euclidean(self, a, b):
        ''' Helper, calculates point to point distance '''
        return np.power(np.power(a[0]-b[0], 2) + np.power(a[1]-b[1], 2), 0.5)
    
    def resetAngles(self):
        ''' Reset angle params '''
        self.th_fem = np.array([0., 0., 0.])
        self.th_kne = np.array([0., 0., 0.])
        self.th_ank = np.array([0., 0., 0.])
        self.th_toe = ([0., 0., 0.])    
        self.base = np.array([[0, 0], [0, 0], [0, -9.81]])
    
    def saveload(self, load = 1):
        if load == 1:
            with open('Params2D.dat', 'rb') as f:
                self.inhand = pickle.load(f)
        else:
            with open('Params2D.dat', 'wb') as f:
                pickle.dump(self.inhand, f)
                
    def getTorques(self):
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        t5 = []
        x = []
        for i in range(100):
            x.append(i)
            t1.append(self.torquelist[i][0,0])
            t2.append(self.torquelist[i][0,1])
            t3.append(self.torquelist[i][0,2])
            t4.append(self.torquelist[i][0,3])
            t5.append(self.torquelist[i][0,4])
        plt.plot(x, t2)
        plt.plot(x, t3)
        plt.plot(x, t4)
        plt.plot(x, t5)
        plt.xlabel('Timestep')
        plt.ylabel('Torque (Nm)')
        plt.legend(['Femur','Tibia', 'Ankle', 'Foot'])
        plt.savefig("Torques2D.png")