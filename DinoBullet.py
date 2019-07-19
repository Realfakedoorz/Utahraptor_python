# -*- coding: utf-8 -*-
"""
Utahraptor pybullet interactions

@author: Sam
"""

import numpy as np
import time
from numpy import random as rd
import pickle
import pybullet as pb
import pybullet_data
import threading
import os
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()
import gc
from datetime import datetime

class dino3D():
    def __init__(self):
        ''' pyBullet params '''
        self.T_fixed = 1./240
        self.SHARED_SRV_FILE = r"C:\bullet3\bin\App_PhysicsServer_SharedMemory_vs2010_x64_release.exe"
        self.SHARED_GUI_FILE = r"C:\bullet3\bin\App_PhysicsServer_SharedMemory_GUI_vs2010_x64_release.exe"
        self.init = 0
        self.CPU_cores = 1
        self.botStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.popsize = 0
        self.pool = Pool(processes=self.CPU_cores)
        self.botStartPos = [0,0,1.5]
        self.maxforce = 9000
        self.scale = 1
        self.Disconnect()
        self.vid = []
        self.elites = []
        self.elitesfit = []
        self.epochnum = 0
        self.dynAccOn = 1
        
        # Finite diff for accelerations
        self.prev_vs = [0,0,0,0,0,0,0,0,0]
        self.accs = [0,0,0,0,0,0,0,0,0]
        
        self.prev_rots = [0,0,0,0,0,0,0,0,0]
        self.acc_rots = [0,0,0,0,0,0,0,0,0]
        self.order = 4
        
        #pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        rd.seed()
        
    def help(self):
        string = '''
        Need to fix Fourier series
        Check the function out first in plots
        
        
        '''
        print(string)
        
        ''' Thread scheduler '''
    def runthreads(self, cid):
        threads = []
        t = threading.Thread(target = self.showIm, args = (cid,))
        threads.append(t)
        t.start()
    
    ''' Workers for interfacing with pybullet '''
    def Connect(self, grav = 1, pb_type_connection = pb.SHARED_MEMORY):
        physicsClient = pb.connect(pb_type_connection)
        
        pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        if grav == 1:
            pb.setGravity(0,0,-9.81)
        pb.setPhysicsEngineParameter(fixedTimeStep = self.T_fixed, 
                                     #solverResidualThreshold = 1- 10, 
                                     numSolverIterations = 50, 
                                     numSubSteps = 4)
        
        self.planeId = pb.loadURDF("plane100.urdf", globalScaling = self.scale)
        
        
        return physicsClient

    def Step(self, steps = 10000, sleep = 1, cid=0, botID=0):
        for i in range(0, steps):
            pb.stepSimulation(cid)
            
            if sleep == 1:
                time.sleep(self.T_fixed)
            self.dynAcc(botID, cid)
            
    def AdjustCamera(self, dist = 3, botID=0, cid=0):
        botPos, botOrn = pb.getBasePositionAndOrientation(botID)
        pb.resetDebugVisualizerCamera( cameraDistance=dist*self.scale, 
                                          cameraYaw=0.0, 
                                          cameraPitch=-20.0, 
                                          cameraTargetPosition=botPos,
                                          physicsClientId = cid)
    
    def Disconnect(self):
        for i in range(100):
            try:
                pb.disconnect(i)
            except:
                pass
        
    def RunSRV(self, GUI=1):
        if GUI == 1:
            os.startfile(self.SHARED_GUI_FILE)
        else:
            os.startfile(self.SHARED_SRV_FILE)
    
    def Init(self,botStartOrientation, pb_type = pb.DIRECT):
        cid = self.Connect(grav=1, pb_type_connection = pb_type)
        # Reset acceleration variables
        self.prev_vs = [0,0,0,0,0,0,0,0,0]
        self.accs = [0,0,0,0,0,0,0,0,0]
        
        self.prev_rots = [0,0,0,0,0,0,0,0,0]
        self.acc_rots = [0,0,0,0,0,0,0,0,0]
        
        #time.sleep(0.1)
        
        botID = pb.loadURDF("./Model/FootUtahBody.SLDASM.urdf",self.botStartPos, #./Model/FootUtahBody.SLDASM.urdf ./Model/Repaired/urdf/Repaired.urdf
                                 botStartOrientation, physicsClientId = cid,
                                 globalScaling = self.scale)
        numJoints = pb.getNumJoints(botID)
        
        for i in range(8): # Disable self collision
            for j in range(8):
                if j != i:
                    pb.setCollisionFilterPair(bodyUniqueIdA = botID, bodyUniqueIdB = botID, 
                                              linkIndexA = i, linkIndexB = j, physicsClientId = cid,
                                              enableCollision = 0)
        
        for i in range(0,numJoints):
            pb.setJointMotorControl2(bodyUniqueId = botID, jointIndex = i, controlMode = pb.POSITION_CONTROL, force = 0)
        
        pb.changeDynamics(bodyUniqueId = 0, linkIndex = -1, lateralFriction = 0.9)
        
        return cid, botID
    
    def setLegs(self, angles, sleep = 0, botID = 0, cid = 0):
        for i in range(8):
            pb.resetJointState(botID, i, targetValue = ((angles[i])), physicsClientId = cid)
                
            if sleep != 0:
                time.sleep(sleep)
      
        
    ''' Standing GA '''
    def selectionStanding(self, select_n = 20):
        ''' Roulette wheel and crossover '''
        pop_tmp = self.population
        fit_tmp = self.fitnesses
        self.fittotal = 0
        
        fitness = self.fitnesses-min(self.fitnesses)+1
        
        for i in range(0,self.popsize):
            self.fittotal += fitness[i]
        
        probabilities = np.round(fitness/self.fittotal, decimals = 3)
        # Fudge to make sum(p) = 1
        if np.sum(probabilities != 1):            
            n = 0
            s = np.sum(probabilities)
            while probabilities[n] < s-1:
                n += 1
                
            probabilities[n] += 1-np.sum(probabilities)
        
        selections = np.ones([select_n])*999
        
        
        for i in range(0,select_n):
            choose = rd.choice(range(0, len(self.population)), p = np.ndarray.tolist(probabilities[:]))
            limit = 0
            while any(selections == float(choose)):
                limit += 1
                choose = rd.choice(range(0, len(self.population)), p = np.ndarray.tolist(probabilities[:]))
                if limit == 20:
                    print("break")
                    limit = 0
                    break
            selections[i] = int(choose)

        #self.generate_population()
        
        for i in range(0, select_n):
            self.population[i] = pop_tmp[int(selections[i])]
            self.fitnesses[i] = fit_tmp[int(selections[i])]
        #print("\nIn select, {:.3f}".format(self.elites[-1][0][0]))
    
        '''for i in range(0,self.popsize):
            self.fitnesses[i] = self.fitnessStanding(i); self.fittotal += self.fitnesses[i,0]'''
        
    def fitnessStanding(self, popNum = 0):
        angles = self.population[popNum][0]
        
        botStartOrientation = pb.getQuaternionFromEuler([0,self.population[popNum][1],0])
        
        simID, botId = self.Init(botStartOrientation, pb_type = pb.DIRECT)
        
        self.setLegs(angles, sleep = 0, botID = botId, cid = simID)
        
        botPos, botOrn = pb.getBasePositionAndOrientation(botId)
        
        
        footOrn = pb.getEulerFromQuaternion(pb.getLinkState(bodyUniqueId = botId, linkIndex = 3, physicsClientId = simID)[1])[1]
        dur = 0
        torpen = 0
        basePosPen = 0
        baseHeightPen = 0
        #print(".")
        #Check x of bot within footprint
        if (pb.getLinkState(bodyUniqueId = botId, linkIndex=3, physicsClientId = simID)[0][0] > pb.getDynamicsInfo(bodyUniqueId = botId, linkIndex = -1, physicsClientId = simID)[3][0]+0.2  
        or pb.getLinkState(bodyUniqueId = botId, linkIndex=3, physicsClientId = simID)[0][0] < pb.getDynamicsInfo(bodyUniqueId = botId, linkIndex = -1, physicsClientId = simID)[3][0]-0.2):
            basePosPen -=10000
            #print("p1")
            
        if pb.getLinkState(bodyUniqueId = botId, linkIndex=3, physicsClientId = simID)[0][2] > pb.getLinkState(bodyUniqueId = botId, linkIndex=2, physicsClientId = simID)[0][2]:
            basePosPen -=30000
            #print("p2")
        
        if pb.getLinkState(bodyUniqueId = botId, linkIndex=2, physicsClientId = simID)[0][2] > pb.getLinkState(bodyUniqueId = botId, linkIndex=1, physicsClientId = simID)[0][2]:
            basePosPen -=10000
            #print("p3")
        
        if pb.getLinkState(bodyUniqueId = botId, linkIndex=1, physicsClientId = simID)[0][2] > botPos[2]:
            basePosPen -=10000
            #print("p4")
        
        if len(angles) < 9:
            angles.append(0)
            
                 
        while botPos[2] > 0.5:
            self.Step(steps = 1, sleep = 0, cid=simID, botID=botId)
            
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                                 targetPositions=angles, 
                                                 targetVelocities = [0]*9,
                                                 forces = [99999999999]*9,
                                                 physicsClientId = simID)
            botPos, botOrn = pb.getBasePositionAndOrientation(botId)
            
            torques = []
            for j in range(8):
                tmp, tmp, tmp, t = pb.getJointState(botId, j)
                torques.append(t)
                if t > 200:
                    torpen += 1

            dur += 1
            if dur == 2000:
                break
        if (pb.getLinkState(bodyUniqueId = botId, linkIndex=3, physicsClientId = simID)[0][0] < pb.getDynamicsInfo(bodyUniqueId = botId, linkIndex = -1, physicsClientId = simID)[3][0]+0.2  
        and pb.getLinkState(bodyUniqueId = botId, linkIndex=3, physicsClientId = simID)[0][0] > pb.getDynamicsInfo(bodyUniqueId = botId, linkIndex = -1, physicsClientId = simID)[3][0]-0.2):
            basePosPen +=50000
            #print("p1")
        footOrn = pb.getEulerFromQuaternion(pb.getLinkState(bodyUniqueId = botId, linkIndex = 3, physicsClientId = simID)[1])[1]    
        if footOrn < 0.2 and footOrn > -0.2:
            footFit = 100000
        else:
            footFit = 0
            
        torques = []
        for j in range(8):
            tmp, tmp, tmp, t = pb.getJointState(1, j)
            torques.append(t)
            
        torqueFit = (150-max(np.abs(torques)))#; print(max(np.abs(torques)))
        
        fit = 50000010+dur*400+botPos[2]*100000 + basePosPen + baseHeightPen - np.linalg.norm(pb.getBaseVelocity(1)[0])*100000 + footFit + 100*torqueFit#-torpen*100
        self.fitnesses[popNum] = fit
        
        if self.fitnesses[popNum] > self.bestfit:
                    self.bestfit = self.fitnesses[popNum]
                    self.best = self.population[popNum]
                    
                    self.elites.append(self.population[popNum])
                    self.elitesfit.append(self.fitnesses[popNum])
                    print("New best fitness {:.0f}".format(self.bestfit))
                    #print("\nIn fit, {:.3f}".format(self.elites[-1][0][0]))
                    with open('elites.dat', 'wb') as f:
                        pickle.dump([self.elites], f)
                    self.showIm(simID, botID = botId)
                    #print(footOrn)
        '''
        if fit > self.bestfit:
            self.bestfit = fit
            self.best2 = self.best
            self.best = self.population[popNum]
            self.elites.append(self.population[popNum])
            self.elitesfit.append(self.fitnesses[popNum])
            print("New best fitness {:.0f}".format(fit))
            print("\nIn fit, {:.3f}".format(self.elites[-1][0][0]))
            #self.showIm(simID)
            #print(footOrn)'''
        
        pb.disconnect(physicsClientId = simID)
        
    def blendStanding(self, select_n = 20):
        sort = np.argsort(-self.fitnesses)
        n = select_n
        tmp = self.elites
        while n < self.popsize:
            
            parent1 = self.population[sort[(0+n)%select_n]]
            parent2 = self.population[sort[(1+n)%select_n]]
            for i in range(0, len(self.population[0][0])):
                d = abs(parent1[0][i]-parent2[0][i])
                self.population[n][0][i] = min([parent1[0][i],parent2[0][i]])+rd.random()*d
                d = abs(parent1[1]-parent2[1])
                self.population[n][1] = min([parent1[1],parent2[1]])+rd.random()*d
            n+=1
        #print("\nIn blend, {:.3f}".format(self.elites[-1][0][0]))
        
    def mutateStanding(self, rate = 0.01):
        #print("\nIn mutate1, {:.3f}".format(self.elites[-1][0][0]))
        for i in range(0,self.popsize):
            if rd.rand() < rate:
                self.population[i][0][0] = rd.randint(-90,0)*np.pi/180
            if rd.rand() < rate:
                self.population[i][0][1] = rd.randint(-120,00)*np.pi/180
            if rd.rand() < rate:
                self.population[i][0][2] = rd.randint(180,280)*np.pi/180
            if rd.rand() < rate:
                self.population[i][0][3] = rd.randint(60,180)*np.pi/180
                
            self.population[i][0][4] = self.population[i][0][0]
            for w in range(5,8):
                self.population[i][0][w] = -self.population[i][0][w-4]
                
            self.population[i][1] -= self.population[i][0][0]
        '''print("\nIn mutate, {:.3f}".format(self.elites[-1][0][0]))'''
                        
    def generateStanding(self):
        class angles:
            def __init__(self): 
                self.theta = []
                
                self.theta.append(rd.randint(-90,0))
                self.theta.append(rd.randint(-120,00))
                self.theta.append(rd.randint(180,280))
                self.theta.append(rd.randint(60,180))
                
                for i in range(4):
                    self.theta.append(self.theta[i])
                    
                self.theta2 = []
                for i in range(5):
                    self.theta2.append(self.theta[i]*np.pi/180)
                for i in range(5,8):
                    self.theta2.append(-self.theta[i]*np.pi/180)
                    
        angle_return = []
        for i in range(self.popsize):
            tmp = angles()
            baseOrn = rd.randint(-90,90)*np.pi/180
            tmp.theta2[0] += baseOrn
            tmp.theta2[4] += baseOrn
            angle_return.append([tmp.theta2,baseOrn])
        return angle_return
         
    def showBestStanding(self, num = -1, log = 0):
        self.Disconnect()
        self.RunSRV(1)
        time.sleep(1)
        now = datetime.now()
        '''print("\nInShow, {:.3f}".format(self.elites[-1][0][0]))'''
        #angles = np.array([90-38+30, 180-83, 180-110, 63, 90-38+30, -(180-83), -(180-110), -63])*np.pi/180#self.population[sort[0]][0]
        #botStartOrientation = pb.getQuaternionFromEuler([0,-30*np.pi/180,0])
        
        #angles = np.array([90-38, 180-83, 180-110, 63, 90-38, -(180-83), -(180-110), -63])*np.pi/180
        #botStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        
        #sort = np.argsort(-self.fitnesses)
        #angles = self.population[sort[0]][0]
        #botStartOrientation = pb.getQuaternionFromEuler([0, self.population[sort[0]][1], 0])
        
        angles = self.elites[num][0]
        botStartOrientation = pb.getQuaternionFromEuler([0, self.elites[num][1], 0])
        
        cid, botId = self.Init(botStartOrientation, pb_type = pb.SHARED_MEMORY)
        if log == 1:
            logID = pb.startStateLogging(loggingType=pb.STATE_LOGGING_VIDEO_MP4, fileName = "stand_{}.mp4".format(now.strftime("%m%d%Y_%H.%M.%S")))  
        print([cid,botId])
        self.setLegs(angles, sleep = 0, botID = botId, cid = cid)
        
        self.Torques = []
        
        for dur in range(0,1000):
            self.Step(1, sleep = 1, cid=cid, botID=botId)
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [99999999999]*9)
            
            torques = []
            for j in range(8):
                tmp, tmp, tmp, t = pb.getJointState(botId, j)
                torques.append(t)
            self.Torques.append(torques)
        if log == 1:
            pb.stopStateLogging(logID)
        
       
    def StandingGA(self, epochs = 10):
        self.epochnum = 0
        
        self.Disconnect()
        tmp = 0
        
        # Initialise server
        #self.runthreads([self.RunSRV, self.Connect])
        if self.popsize == 0:
            self.popsize = 50
            self.fitnesses = np.ones(self.popsize)
            self.population = self.generateStanding()

            self.bestfit = 0
            self.best = self.population[0]

        for ep in range(epochs):
            self.epochnum = ep
            print("Epoch {:d}".format(ep))
            
            
            for i in range(self.popsize):
                self.fitnessStanding(i)
                
                #print(".",end="")
            #tracker.print_diff()        
            self.selectionStanding()
            #print("\nselect, {:.3f}".format(self.elites[-1][0][0]))
            gc.collect()
            #print(rd.randint(100))
            #tracker.print_diff()
            gc.collect()
            self.blendStanding()
            #print("\nblend, {:.3f}".format(self.elites[-1][0][0]))
            gc.collect()
            #tracker.print_diff()
            self.mutateStanding(rate=max([0.5*(1-ep/(0.7*epochs)), 0.01]))
            with open('elites.dat', 'rb') as f:
                        [self.elites] = pickle.load(f)
            #print("\nIn fit, {:.3f}".format(self.elites[-1][0][0]))
            #tracker.print_diff()
            gc.collect()
            
        self.Disconnect()
        self.showBestStanding()
    
    
    
    
    
    
    
    
    
    '''Add to walking, multiple amplitudes and time constants - allows less linear motion '''
    
    
    
    
    
    
    
    ''' Onto moving '''
    def selectionWalk(self, select_n = 15):
        ''' Roulette wheel and crossover '''
        pop_tmp = self.population
        popoff_tmp = self.popBodyOffsets
        fit_tmp = self.fitnesses
        
        self.fittotal = 0
        
        for i in range(0,self.popsize):
            self.fittotal += self.fitnesses[i]
        
        probabilities = np.round(self.fitnesses/self.fittotal, decimals = 7)
        # Fudge to make sum(p) = 1
        if np.sum(probabilities != 1):            
            n = 0
            s = np.sum(probabilities)
            while probabilities[n] < s-1:
                n += 1
                
            probabilities[n] += 1-np.sum(probabilities)
        
        selections = np.ones([select_n])*999
        
        
        for i in range(0,select_n):
            choose = rd.choice(range(0, len(self.population)), p = np.ndarray.tolist(probabilities[:]))
            limit = 0
            while any(selections == float(choose)):
                limit += 1
                choose = rd.choice(range(0, len(self.population)), p = np.ndarray.tolist(probabilities[:]))
                if limit == 20:
                    print("break")
                    limit = 0
                    break
            selections[i] = int(choose)

        #self.generate_population()
        for i in range(0, select_n):
            self.population[i] = pop_tmp[int(selections[i])]
            self.popBodyOffsets[i] = popoff_tmp[int(selections[i])]
            self.fitnesses[i] = fit_tmp[int(selections[i])]
    
        '''for i in range(0,self.popsize):
            self.fitnesses[i] = self.fitnessWalk(i); self.fittotal += self.fitnesses[i,0]'''
               
    def blendWalk(self, select_n = 15):
        ''' Blend (BLX-alpha) best parents up to generate the children '''
        sort = np.argsort(-self.fitnesses)
        
        n = select_n
        
        while n < self.popsize:
            parent1 = self.population[sort[(0+n)%select_n]]; p1off = self.popBodyOffsets[sort[(0+n)%select_n]]
            parent2 = self.population[sort[(1+n)%select_n]]; p2off = self.popBodyOffsets[sort[(1+n)%select_n]]
            
            for j in range(0, len(self.population[0,:])):
                for k in range(0, len(self.population[0,0,:])):
                    d = abs(parent1[j,k]-parent2[j,k])
                    
                    self.population[n,j,k] = min([parent1[j,k],parent2[j,k]])+rd.random()*d
            
            d = abs(p1off-p2off)
            self.popBodyOffsets[n] = min([p1off, p2off])+rd.random()*d
            
            n+=1
            n+=1
        
    def mutateWalk_Old(self, rate = 0.01):
        for i in range(0,self.popsize):
            for w in range(0, len(self.population[0,0,:])):
                for t in range(4):
                    '''t_l = [-26.,  -17.,  194.,  122.]
                    if rd.rand() < rate:
                        if t!=0:
                            self.population[i, 0, t] = rd.randint(-5,5)            #a0
                        else:
                            self.population[i, 0, t] = rd.randint(-10,0)
                        self.population[i, 0, t] += t_l[t]
                #    self.population[i, 0, w] = rd.randint(0,360) #theta0xs'''
                    
                if rd.rand() < rate:
                    self.population[i, 1, w] = rd.rand()*100-50#amplitudes
                    
                if rd.rand() < rate:
                    self.population[i, 2, w] = 2*rd.rand()-1#Ts
                
                #Second sine parameters
                if rd.rand() < rate:
                    self.population[i, 3, w] = rd.rand()*100-50
                
                if rd.rand() < rate:
                    self.population[i, 4, w] = 2*rd.rand()-1#Ts
                    
                if rd.rand() < rate:
                    self.population[i, 5, w] = rd.rand()*100-50
                
                if rd.rand() < rate:
                    self.population[i, 6, w] = 2*rd.rand()-1#Ts
                    
                if rd.rand() < rate:
                    self.population[i, 7, w] = rd.rand()*100-50
                
                if rd.rand() < rate:
                    self.population[i, 8, w] = 2*rd.rand()-1#Ts
            #if rd.rand() < rate:
                #self.popBodyOffsets[i] = rd.randint(0,360)
                
    def mutateWalk(self, rate = 0.01):
        for i in range(0,self.popsize):
            for w in range(0, len(self.population[0,0,:])):
                for t in range(4):
                    '''t_l = [-26.,  -17.,  194.,  122.]
                    if rd.rand() < rate:
                        if t!=0:
                            self.population[i, 0, t] = rd.randint(-5,5)            #a0
                        else:
                            self.population[i, 0, t] = rd.randint(-10,0)
                        self.population[i, 0, t] += t_l[t]
                #    self.population[i, 0, w] = rd.randint(0,360) #theta0xs'''
                    
                if rd.rand() < rate:
                    self.population[i, 1, w] += rd.rand()*10-5#amplitudes
                    
                if rd.rand() < rate:
                    self.population[i, 2, w] += 0.2*rd.rand()-0.1#Ts
                    if abs(self.population[i, 2, w]) > 1: self.population[i, 2, w] = 1*np.sign(self.population[i, 2, w])
                
                #Second sine parameters
                if rd.rand() < rate:
                    self.population[i, 3, w] += rd.rand()*10-5
                
                if rd.rand() < rate:
                    self.population[i, 4, w] += 0.2*rd.rand()-0.1#Ts
                    if abs(self.population[i, 4, w]) > 1: self.population[i, 4, w] = 1*np.sign(self.population[i, 4, w])

                if rd.rand() < rate:
                    self.population[i, 5, w] += rd.rand()*10-5
                
                if rd.rand() < rate:
                    self.population[i, 6, w] += 0.2*rd.rand()-0.1#Ts
                    if abs(self.population[i, 6, w]) > 1: self.population[i, 6, w] = 1*np.sign(self.population[i, 6, w])

                if rd.rand() < rate:
                    self.population[i, 7, w] += rd.rand()*10-5
                
                if rd.rand() < rate:
                    self.population[i, 8, w] += 0.2*rd.rand()-0.1#Ts
                    if abs(self.population[i, 8, w]) > 1: self.population[i, 8, w] = 1*np.sign(self.population[i, 8, w])
        
            #if rd.rand() < rate:
                #self.popBodyOffsets[i] = rd.randint(0,360)         
                        
    def generateWalk(self):
        ''' Generate 20 random sets of parameters '''
        self.population = np.zeros([self.popsize,9,4])
        self.popBodyOffsets = np.zeros(self.popsize)
        rd.seed()
        for d in range(0,self.popsize):
            self.population[d, 0] = rd.randint(-5,5, size=4)            #a0
            self.population[d, 0] += [-26.,  -17.,  194.,  122.]
            a = rd.randint(-10,0)
            self.population[d, 0, 1] = a-26
            for w in range(0,4):
                self.population[d, 1, w] = rd.rand()*100-50          #a1 1st amplitude
                self.population[d, 2, w] = 2*rd.rand()-1  #T1 1st phase shift
                self.population[d, 3, w] = rd.rand()*100-50            #a2 2nd amplitude
                self.population[d, 4, w] = 2*rd.rand()-1  #T2 2nd phase shift
                self.population[d, 5, w] = rd.rand()*100-50            #a2 2nd amplitude
                self.population[d, 6, w] = 2*rd.rand()-1  #T2 2nd phase shift
                self.population[d, 7, w] = rd.rand()*100-50           #a2 2nd amplitude
                self.population[d, 8, w] = 2*rd.rand()-1  #T2 2nd phase shift
            
            self.popBodyOffsets[d] = rd.randint(0,360)
            self.population[d,0,:] = [-26.,  -17.,  194.,  122.]
      
    def fitnessWalk(self, popNum = 0):
        self.dynAccOn = 1
        legt0   = self.population[popNum, 0]
        legamp  = self.population[popNum, 1]
        legT    = self.population[popNum, 2]
        legamp2 = self.population[popNum, 3]
        legT2   = self.population[popNum, 4]
        legamp3 = self.population[popNum, 5]
        legT3   = self.population[popNum, 6]
        legamp4 = self.population[popNum, 7]
        legT4   = self.population[popNum, 8]
        
        #offSet = self.popBodyOffsets[popNum]
        T = 0.7
        
        simID,botId = self.setStanding(simtype = pb.DIRECT, num=6)
        
        botPos, botOrn = pb.getBasePositionAndOrientation(botId)
        footLoc = pb.getLinkState(botId, linkIndex=3)[0][2]
        
        '''Deprecate this attempt at ramping? Seems to be sorted by velocity limits '''
        walkbase = np.array([-26.,  -17.,  194.,  122., -26., 17., -194., -122, 0])
        angs = np.array([legt0[0], legt0[1], legt0[2], legt0[3], legt0[0], -legt0[1], -legt0[2], -legt0[3], 0])
        diff = angs-walkbase
        
        dur = 0
        pencon = 0
        speed = 0
        sc = 1
        OrnFit = 0
        ZMPdist = 0
        
        '''NOTE: Right, fuck this, time to get the feet flat and the tail working ''' 
        
        
        ''' Modify the fkine function for this '''
        T = abs(1.5/(2*self.fkine([ legt0[0]+legamp[0]+legamp2[0]+legamp3[0], legt0[1]+legamp[1]+legamp2[1]+legamp3[1], legt0[2]+legamp[2]+legamp2[2]+legamp3[2], legt0[3]+legamp[3]+legamp2[3]+legamp3[3] ])[0]/1000))
        if T>3.5: T=3.5
        lin_vel, ang_vel= pb.getBaseVelocity(botId)
        
        
        while pb.getBaseVelocity(botId)[0][1] < 0:
            self.Step(steps = 1, sleep = 0, cid=simID, botID=botId)
        
        for i in range(4000):
            if botPos[2] > 0.55 and botPos[2] < 1.6 and footLoc < 0.4 and footLoc < botPos[2]:
                dur += 1
                t = float(i)*(self.T_fixed)
                
                #legamp[0]*np.sin(legamp[0]) + sc*legamp2[]
                
#                if t < T: #ramp up angle
#                    angles = walkbase + (i*T/self.T_fixed)*diff
#                else:
                
                angles = self.f4(popNum, t, T, el=0)
                
                #a_foot = (pb.getEulerFromQuaternion(OrnInit)[1]+angles[0]+angles[1]+angles[2]-53*np.pi/180)
                #angles[3] = a_foot
                #angles[7] = -a_foot
                
        
                pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                                 targetPositions=angles, 
                                                 targetVelocities = [0]*9,
                                                 forces = [self.maxforce]*9,
                                                 physicsClientId = simID)
                    #self.setLegs(angles, sleep = 0, botID = botId, cid = simID)
                    
                self.Step(steps = 1, sleep = 0, cid=simID, botID=botId)
                ZMP = self.ZMP(botId, simID)
                
                footLoc = pb.getLinkState(botId, linkIndex=3)[0][2]
                
                ZMP = self.ZMP(botId, simID)
                footLoc1 = pb.getLinkState(botId, linkIndex=3)[0][0:2]
                footLoc2 = pb.getLinkState(botId, linkIndex=7)[0][0:2]
                footPrint = [(footLoc1[0]+footLoc2[0])/2,  (footLoc1[1]+footLoc2[1])/2]
                ZMPdist += np.sqrt((ZMP[0]-footPrint[0])**2 + (ZMP[1]-footPrint[1])**2)
                
                
                botPos, botOrn = pb.getBasePositionAndOrientation(botId)
                lin_vel, ang_vel= pb.getBaseVelocity(botId) 
                speed = lin_vel[0]
                '''torques = []
                for j in range(8):
                    tmp, tmp, tmp, t = pb.getJointState(botId, j)
                    torques.append(t)
                    if t > 5000:
                        torpen += 1'''
                pencon += self.penConsts(cid = simID, botID = botId)

                botAngles = np.array(pb.getEulerFromQuaternion(botOrn))%(2*np.pi)*180/np.pi
                if botAngles[0] < 150 and botAngles[0] > 30:
                    OrnFit = 0#100000
                else:
                    OrnFit = 0#-10000
            
                if botAngles[2] > 270 or botAngles[2] < 90:
                    OrnFit += 1
                    
        footHeight = pb.getLinkState(bodyUniqueId = botId, linkIndex = 3, physicsClientId = simID)[0][2]
        fit = (t/self.T_fixed)*10000 + botPos[0] * 250000 + speed*10000 + OrnFit*100 - ZMPdist*100000#- 10000*botPos[1]
        if fit > 0:
            self.fitnesses[popNum] = fit
        else:
            self.fitnesses[popNum] = rd.randint(0,3)
        if fit > self.bestfit:
            self.bestfit = fit
            self.best = self.population[popNum]
            print("New best : {:.0f}".format(fit))
            print("Managed {dur:f} seconds, T = {step:f}".format(dur=t, step=T))
            self.elites.append(self.population[popNum])
            #self.showIm(simID, botID = botId)
            with open('elites.dat', 'wb') as f:
                        pickle.dump([self.elites], f)
        pb.disconnect(physicsClientId = simID)    
    
    def WalkingGA(self, epochs = 10, select_n = 15):
        self.Disconnect()
        # Initialise server
        #self.runthreads([self.RunSRV, self.Connect])
        if self.popsize == 0:
            self.popsize = 60
            self.fitnesses = np.zeros(self.popsize)
            self.population = np.zeros([self.popsize,5,4])
            self.popBodyOffsets = np.zeros(self.popsize)
             
            self.OffSet = 0
            self.generateWalk()
            self.bestfit = 0
            self.best = self.population[0]
        
        for ep in range(epochs):
            print("Epoch {:d}".format(ep))
            
            for i in range(self.popsize):
                self.fitnessWalk(i)
            gc.collect()
            self.selectionWalk(select_n)
            gc.collect()
            self.blendWalk(select_n)
            gc.collect()
            self.mutateWalk(rate=max([0.2*(1-ep/(0.7*epochs)), 0.01]))
            with open('elites.dat', 'rb') as f:
                        [self.elites] = pickle.load(f)
            gc.collect()
            
        pb.disconnect()
    
    def showBestWalk(self, log = 0, num = -1):
        self.Disconnect()
        self.RunSRV(1)
        time.sleep(1)
        now = datetime.now()
        self.dynAccOn = 0
        
        legt0 = self.elites[num][0]
        legamp = self.elites[num][1]
        legT = self.elites[num][2]
        legamp2 = self.elites[num][3]
        legT2   = self.elites[num][4]
        legamp3 = self.elites[num][5]
        legT3 = self.elites[num][6]
        legamp4 = self.elites[num][7]
        legT4 = self.elites[num][8]
        
        #offSet = self.popBodyOffsets[popNum]
        T = 0.5
        
        walkbase = np.array([-26.,  -17.,  194.,  122., -26., 17., -194., -122, 0])
        angs = np.array([legt0[0], legt0[1], legt0[2], legt0[3], legt0[0], -legt0[1], -legt0[2], -legt0[3], 0])
        diff = angs-walkbase
        
        simID,botId = self.setStanding(simtype = pb.SHARED_MEMORY, num=6)#simID, botId = self.setStanding(simtype = pb.DIRECT, num = 3, anglesset = angles, height = heightDiff)
        
        botPos, botOrn = pb.getBasePositionAndOrientation(botId)
        footLoc = pb.getLinkState(botId, linkIndex=3)[0][2]
        heightDiff = botPos[2]-footLoc
        sc = 1
        OrnInit = botOrn
        
        T = abs(1.5/(2*self.fkine([ legt0[0]+legamp[0]+legamp2[0]+legamp3[0], legt0[1]+legamp[1]+legamp2[1]+legamp3[1], legt0[2]+legamp[2]+legamp2[2]+legamp3[2], legt0[3]+legamp[3]+legamp2[3]+legamp3[3] ])[0]/1000))        
        if T>3.5: T=3.5
        
        self.Torques = []
        
        while pb.getBaseVelocity(botId)[0][1] < 0:
            self.Step(steps = 1, sleep = 0, cid=simID, botID=botId)
            print(pb.getBaseVelocity(botId)[0][1])
        
        for i in range(10000):
            t = float(i)*(self.T_fixed)
            
                        
            angles = self.f4(num, t, T)
            
            #a_foot = (pb.getEulerFromQuaternion(OrnInit)[1]+angles[0]+angles[1]+angles[2]-53*np.pi/180)
            #angles[3] = a_foot
            #angles[7] = -a_foot
                              
            if i == 0:
                if log == 1:
                    logID = pb.startStateLogging(loggingType=pb.STATE_LOGGING_VIDEO_MP4, fileName = "walk_{}.mp4".format(now.strftime("%m%d%Y_%H.%M.%S")))  
                    
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                             targetPositions=angles, 
                                             targetVelocities = [0]*9,
                                             forces = [self.maxforce]*9,
                                             physicsClientId = simID)
            
            self.Step(steps = 1, sleep = 1, cid=simID, botID=botId)
            self.AdjustCamera(botID = botId, cid = simID)
            
            torques = []
            for j in range(9):
                tmp, tmp, tmp, t = pb.getJointState(botId, j)
                torques.append(t)
            self.Torques.append(torques)
            
            botPos, botOrn = pb.getBasePositionAndOrientation(botId)
            footLoc = pb.getLinkState(botId, linkIndex=3)[0][2]
            if botPos[2] < 0.55 or botPos[2] > 2 or footLoc > botPos[2]:
                break
            #print(botPos)
        if log == 1:    
            pb.stopStateLogging(loggingId = logID)
    
    
    
    def f4(self, num, t, T, el=1):
        if el == 1:
            legt0 = self.elites[num][0]
            legamp = self.elites[num][1]
            legT = self.elites[num][2]
            legamp2 = self.elites[num][3]
            legT2   = self.elites[num][4]
            legamp3 = self.elites[num][5]
            legT3   = self.elites[num][6]
            legamp4 = self.elites[num][7]
            legT4   = self.elites[num][8]
        else:
            legt0 = self.population[num][0]
            legamp = self.population[num][1]
            legT = self.population[num][2]
            legamp2 = self.population[num][3]
            legT2   = self.population[num][4]
            legamp3 = self.population[num][5]
            legT3   = self.population[num][6]
            legamp4 = self.population[num][7]
            legT4   = self.population[num][8]
    
        f0 = np.array([legt0[0], legt0[1], legt0[2], legt0[3], legt0[0], -legt0[1], -legt0[2], -legt0[3], 0])
        
        f1 = np.array([legamp[0]*np.sin((2*np.pi/T)*(t - legT[0])), legamp[1]*np.sin((2*np.pi/T)*(t - legT[1])),
           legamp[2]*np.sin((2*np.pi/T)*(t - legT[2])), legamp[3]*np.sin((2*np.pi/T)*(t - legT[3])),
           legamp[0]*np.sin((3*np.pi/T)*(np.pi+ t - legT[0])), -legamp[1]*np.sin((3*np.pi/T)*(np.pi+ t - legT[1])),
           -legamp[2]*np.sin((3*np.pi/T)*(np.pi+ t - legT[2])), -legamp[3]*np.sin((3*np.pi/T)*(np.pi+ t - legT[3])),
           0])
    
        f2 = np.array([legamp2[0]*(np.sin((2*np.pi/T)*(2*t - legT2[0]))), legamp2[1]*(np.sin((2*np.pi/T)*(2*t - legT2[1]))),
           legamp2[2]*(np.sin((2*np.pi/T)*(2*t - legT2[2]))),             legamp2[3]*(np.sin((2*np.pi/T)*(2*t - legT2[3]))),
           legamp2[0]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT2[0]))),     -legamp2[1]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT2[1]))),
           -legamp2[2]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT2[2]))),    -legamp2[3]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT2[3]))),
           0])
    
        f3 = np.array([legamp3[0]*(np.sin((2*np.pi/T)*(2*t - legT3[0]))), legamp3[1]*(np.sin((2*np.pi/T)*(2*t - legT3[1]))),
           legamp3[2]*(np.sin((2*np.pi/T)*(2*t - legT3[2]))),             legamp3[3]*(np.sin((2*np.pi/T)*(2*t - legT3[3]))),
           legamp3[0]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT3[0]))),     -legamp3[1]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT3[1]))),
           -legamp3[2]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT3[2]))),    -legamp3[3]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT3[3]))),
           0])
        
        f4 = np.array([legamp4[0]*(np.sin((2*np.pi/T)*(2*t - legT4[0]))), legamp4[1]*(np.sin((2*np.pi/T)*(2*t - legT4[1]))),
           legamp4[2]*(np.sin((2*np.pi/T)*(2*t - legT4[2]))),             legamp4[3]*(np.sin((2*np.pi/T)*(2*t - legT4[3]))),
           legamp4[0]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT4[0]))),     -legamp4[1]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT4[1]))),
           -legamp4[2]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT4[2]))),    -legamp4[3]*(np.sin((3*np.pi/T)*(np.pi+ 2*t - legT4[3]))),
           0])
        
        if self.order == 1:
            angles = (f0+f1)*np.pi/180
        elif self.order == 2:
            angles = (f0+f1+f2)*np.pi/180
        elif self.order == 3:
            angles = (f0+f1+f2+f3)*np.pi/180
        elif self.order == 4:
            angles = (f0+f1+f2+f3+f4)*np.pi/180
            
        return angles
    
    def ZMP(self, botID, simID):
        g= 9.81
        Px = 0
        Py = 0
        denom = 0
        
        for i in range(9):
            m_l     = pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = i-1, physicsClientId = simID)[0]
            I       = pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = i-1, physicsClientId = simID)[2]
            if i == 0:
                pos     = pb.getBasePositionAndOrientation(bodyUniqueId = botID, physicsClientId = simID)[0]
            else:
                pos     = pb.getLinkState(bodyUniqueId = botID, linkIndex = i-1, physicsClientId = simID)[0]
            acc_lin = self.accs[i]
            acc_rot = self.acc_rots[i]
            Px      += (m_l*(acc_lin[2]+g)*pos[0] - m_l*acc_lin[0]*pos[2] - I[1]*acc_rot[1])
            Py      += (m_l*(acc_lin[2]+g)*pos[1] - m_l*acc_lin[1]*pos[2] - I[0]*acc_rot[0])
            denom   += (acc_lin[2]+g)*m_l
        
        Px /= denom
        Py /= denom
        return Px, Py
            
    
    def dynAcc(self, botID, simID):
        '''Finite diff for accelerations (1 time step, cba)'''
        prev_vs     = np.array(self.prev_vs, dtype=object)
        accs        = np.array(self.accs, dtype=object)
        prev_rots   = np.array(self.prev_rots, dtype=object)
        acc_rots    = np.array(self.acc_rots, dtype=object)
        
        vs = prev_vs
        rots = prev_rots

        for i in range(9):
            if i == 0:
                state = np.array(pb.getBaseVelocity(bodyUniqueId = botID, physicsClientId = simID))
            else:
                state = np.array(pb.getLinkState(bodyUniqueId = botID, linkIndex = i-1, physicsClientId = simID, computeLinkVelocity=1)[6:8])
            
            vs[i] = np.array(state[0])
            rots[i] = np.array(state[1])
            
            accs[i]         = (vs[i]   - prev_vs[i])/self.T_fixed
            acc_rots[i]     = (rots[i] - prev_rots[i])/self.T_fixed
            prev_vs[i]      = vs[i]
            prev_rots[i]    = rots[i]
        
        self.prev_vs = prev_vs
        self.accs = accs
        
        self.prev_rots = prev_rots
        self.acc_rots = acc_rots
        
    ''' Test etc macros '''
    
    def testLegs(self):
        self.setLegs(np.array([90-38, 180-83, 180-110, 63, 90-38, -(180-83), -(180-110), -63])*np.pi/180)
        
    def saveload(self, load = 1, file = "Angles.dat"):
        if load == 1:
            with open(file, 'rb') as f:
                [self.population, self.fitnesses, self.best, self.bestfit, self.elites, self.elitesfit] = pickle.load(f)
        else:
            with open(file, 'wb') as f:
                pickle.dump([self.population, self.fitnesses, self.best, self.bestfit, self.elites, self.elitesfit], f)

    
    def setStanding(self, simtype = pb.SHARED_MEMORY, num=0, anglesset = 0, height = 0):
        if num== 0:
            self.botStartPos = self.scale*np.array([0.08300925547894937, 0.002430267340262286, 0.8458748281523655])
            botOrn = [3.940743050000044e-05, 0.02249885497625592, -0.004428984037646466, 0.9997370574667152]
            angles = np.array([ 51.96582967,  97.03677617,  70.47829377,  61.78680965,
                               51.95977158, -97.04235016, -70.55186208, -61.87077155, 0])*np.pi/180
            
            cid, botId = self.Init(botOrn, pb_type = simtype)
            self.setLegs(angles, sleep = 0, botID = botId, cid = cid)
            
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [999999]*9,
                                         physicsClientId = cid)
        elif num == 1:
            self.botStartPos = self.scale*np.array([0.041268085601263, 0.099768429873335, 1.1749753510013218])#was 1.2
            botOrn = [0.9813615231606528, 0.0009370188924839653, 0.19216827618774338, -0.00019100374330870658]
            angles = np.array([ 141.77655842,  332.53291582,   47.9706059 ,  159.23242504,
                               141.78012687, -332.53399047,  -47.97753527, -159.23824605, 0])*np.pi/180
            
            cid, botId = self.Init(botOrn, pb_type = simtype)
            self.setLegs(angles, sleep = 0, botID = botId, cid = cid); np.append(angles, 0)
            print(angles)
            
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [999999]*9,
                                         physicsClientId = cid)   
        elif num == 3:
            self.botStartPos = self.scale*np.array([0, 0, height])#was 1.2
            botOrn = [3.940743050000044e-05, 0.02249885497625592, -0.004428984037646466, 0.9997370574667152]
            angles = np.array(anglesset)*np.pi/180
            np.append(angles, 0)
            
            cid, botId = self.Init(botOrn, pb_type = simtype)
            self.setLegs(angles, sleep = 0, botID = botId, cid = cid)
            
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [999999]*9,
                                         physicsClientId = cid)
            pb.resetBaseVelocity(objectUniqueId = botId, physicsClientId = cid, linearVelocity = [0,0,0], angularVelocity= [0,0,0])
        elif num == 5:
            self.botStartPos = self.scale*np.array([-0.005423497042570174, -0.0006093378765514721, 0.9902877456256021])
            botOrn = [-5.9789883568907994e-05,  0.26071283829436115,  0.0001462250556024903,  0.9654163821853766]
            angles = np.array([ -43.        ,  -46.        ,  238.16050597,  154.93573783,
        -43.        ,   46.        , -238.16050597, -154.93573783, 0])*np.pi/180
            
            cid, botId = self.Init(botOrn, pb_type = simtype)
            self.setLegs(angles, sleep = 0, botID = botId, cid = cid)
            
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [999999]*9,
                                         physicsClientId = cid)
        
        elif num == 6:
            self.botStartPos = self.scale*np.array([0.008122401502023714, 0.00022747351081286866, 1.3901162942408427])
            botOrn = [0.00023243435960631722, 0.23408647246703776, -0.0009781480527728955, 0.9722152604277481]
            angles = np.array([ -26.,  -17.,  194.,  122.,  -26.,   17., -194., -122.,    0.])*np.pi/180
            
            cid, botId = self.Init(botOrn, pb_type = simtype)
            self.setLegs(angles, sleep = 0, botID = botId, cid = cid)
            
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [999999]*9,
                                         physicsClientId = cid)
        elif num == 7:
            self.botStartPos = self.scale*np.array([0, 0, height])#was 1.2
            botOrn = [0,0,0,1]
            angles = np.array([3.8,0,133,168,3.8,0,-133,-168,0])*np.pi/180
            np.append(angles, 0)
            
            cid, botId = self.Init(botOrn, pb_type = simtype)
            self.setLegs(angles, sleep = 0, botID = botId, cid = cid)
            
            self.AdjustCamera(botID = botId, cid = cid)
            pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                         targetPositions=angles, 
                                         targetVelocities = [0]*9,
                                         forces = [999999]*9,
                                         physicsClientId = cid)
            pb.resetBaseVelocity(objectUniqueId = botId, physicsClientId = cid, linearVelocity = [0,0,0], angularVelocity= [0,0,0])

        
        self.botStartPos = ([0,0,2])
        return cid, botId
    
    def penConsts(self, cid, botID):
        angles = []
        for i in range(8):
            angles.append(pb.getJointState(bodyUniqueId = botID, jointIndex = i)[0])
        angles = np.array(np.array(angles)*180/np.pi)
        
        penalty = 0
        
        if angles[0] < 290 and angles[0] > 80:
            penalty += 1
            
        #if angles[1]:
            
        # Keep legs down
        if pb.getLinkState(bodyUniqueId = botID, linkIndex = 3, physicsClientId = cid)[0][2] > pb.getLinkState(bodyUniqueId = botID, linkIndex = 1,  physicsClientId = cid)[0][2]:
            penalty += 10
            
        if (pb.getLinkState(bodyUniqueId = botID, linkIndex=3, physicsClientId = cid)[0][0] > pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = -1, physicsClientId = cid)[3][0]+0.2  
        or pb.getLinkState(bodyUniqueId = botID, linkIndex=3, physicsClientId = cid)[0][0] < pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = -1, physicsClientId = cid)[3][0]-0.2):

            penalty -=100
            
        if (pb.getLinkState(bodyUniqueId = botID, linkIndex=3, physicsClientId = cid)[0][2] > pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = -1, physicsClientId = cid)[3][2]
        or pb.getLinkState(bodyUniqueId = botID, linkIndex=3, physicsClientId = cid)[0][2] > pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = 0, physicsClientId = cid)[3][2]
        or pb.getLinkState(bodyUniqueId = botID, linkIndex=3, physicsClientId = cid)[0][2] > pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = 1, physicsClientId = cid)[3][2]
        or pb.getLinkState(bodyUniqueId = botID, linkIndex=3, physicsClientId = cid)[0][2] > pb.getDynamicsInfo(bodyUniqueId = botID, linkIndex = 2, physicsClientId = cid)[3][2]):
            penalty -= 100
        
        return penalty
    
    def showIm(self,cid, show = 0, botID = 0):
        img = [[1, 2, 3] * 50] * 100  #np.random.rand(200, 320)
        
        image = plt.imshow(img, interpolation='none', animated=True, label="blah")
        ax = plt.gca()
        plt.ion()
        
        botPos, botOrn = pb.getBasePositionAndOrientation(botID)
        
        camTargetPos = botPos
        cameraUp = [0, 0, 1]
        cameraPos = [1, 1, 1]
        
        pitch = -20.0
        
        yaw = 0
        roll = 0
        upAxisIndex = 2
        camDistance = 2
        pixelWidth = 640
        pixelHeight = 400
        nearPlane = 0.01
        farPlane = 30
        fov = 60
        viewMatrix = pb.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                            roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight
        projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        img_arr = pb.getCameraImage(pixelWidth,
                                          pixelHeight,
                                          viewMatrix,
                                          projectionMatrix,
                                          shadow=1,
                                          lightDirection=[1, 1, 1],
                                          renderer=pb.ER_BULLET_HARDWARE_OPENGL, physicsClientId = cid)

        w = img_arr[0]  #width of the image, in pixels
        h = img_arr[1]  #height of the image, in pixels
        rgb = img_arr[2]  #color data RGB
        
        np_img_arr = np.reshape(rgb, (h, w, 4))
        np_img_arr = np_img_arr * (1. / 255.)
        self.vid.append(np_img_arr)
        image.set_data(np_img_arr)
        plt.title("# {:d} epoch {:d}".format(len(self.elites)-1, self.epochnum))
        #plt.draw() # draw the plot
        plt.pause(0.5)
        
    def saveVid(self, name = "file.mp4"):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Standing Evolution', artist='',
                        comment='Sam Wilcock')
        writer = FFMpegWriter(fps=0.5, metadata=metadata)
        
        fig = plt.figure()
        img = [[1, 2, 3] * 50] * 100
        l = plt.imshow(img, interpolation='none', animated=True, label="blah")
        self.vid.append(self.vid[-1])
        with writer.saving(fig, name, 100):
            for i in range(len(self.vid)):
                l.set_data(self.vid[i])
                writer.grab_frame()
        self.vid = []
        
    def fkine(self, angles=[0,0,0,0]):
        L_femur = 565
        L_tibia = 505
        L_tarsa = 50
        L_foot = 380
        
        angles[0] += 90
        th = [0, 0]
        for i in angles:
            th.append(i*np.pi/180)
        
        c2 = np.cos(th[2])
        s2 = np.sin(th[2])
        c4 = np.cos(th[4])
        s4 = np.sin(th[4]);
        c23 = np.cos(th[2] + th[3])
        s23 = np.sin(th[2] + th[3])
        c45 = np.cos(th[4] + th[5]) 
        s45 = np.sin(th[4] + th[5])
        
        femur = np.array([L_femur*c2, L_femur*s2])
        tibia = femur+  np.array([L_tibia*c23, L_tibia*s23])
        tarsus = tibia+ np.array([L_tarsa*(c4*c23 - s4*s23), L_tarsa*(c4*s23 + s4*c23)])
        toe = tarsus+   np.array([L_foot*(c23*c45 - s23*s45), L_foot*(c23*s45 + s23*c45)])
        
        return tarsus
    
    
    def setWalkParams(self):
        self.a0 = [-4.30000000e+01, -4.60000000e+01,  2.38160506e+02, 1.54935738e+02]
        self.a1 = [-3.98144313e+01, -9.74788745e+00, -3.60004156e+00, -6.70286657e+00]
        self.T1 = [ 1.00555825e+00,  5.47790755e-02,  2.89667092e-01, 5.11000934e-01]
        self.a2 = [ 2.52137472e+01,  1.59878619e+01,  1.17385969e+01, 1.98134763e+01]
        self.T2 = [ 9.08245134e-01, -2.30661017e-01, -4.74756371e-01, 1.27556671e+00]
        
      
    def demo(self):
        self.RunSRV(1)
        time.sleep(1)
        l = 0
        for num in range(len(self.elites)):
            self.Disconnect()


            T = 0.7
                            
            simID,botId = self.setStanding(simtype = pb.SHARED_MEMORY, num=6)#simID, botId = self.setStanding(simtype = pb.DIRECT, num = 3, anglesset = angles, height = heightDiff)
            
            if l==0:logID = pb.startStateLogging(loggingType=pb.STATE_LOGGING_VIDEO_MP4, fileName = "demo.mp4")
            l = 1
            botPos, botOrn = pb.getBasePositionAndOrientation(botId)
            footLoc = pb.getLinkState(botId, linkIndex=3)[0][2]
            
            self.Torques = []
            
            while pb.getBaseVelocity(botId)[0][1] < 0:
                self.Step(steps = 1, sleep = 0, cid=simID, botID=botId)
                print(pb.getBaseVelocity(botId)[0][1])
            
            for i in range(10000):
                t = float(i)*(self.T_fixed)
                
                            
                angles = self.f4(num, t, T)
                
                pb.setJointMotorControlArray(botId, range(9), controlMode = pb.POSITION_CONTROL, 
                                                 targetPositions=angles, 
                                                 targetVelocities = [0]*9,
                                                 forces = [self.maxforce]*9,
                                                 physicsClientId = simID)
                
                self.Step(steps = 1, sleep = 1, cid=simID, botID=botId)
                self.AdjustCamera(botID = botId, cid = simID)
                
                torques = []
                for j in range(9):
                    tmp, tmp, tmp, t = pb.getJointState(botId, j)
                    torques.append(t)
                self.Torques.append(torques)
                
                botPos, botOrn = pb.getBasePositionAndOrientation(botId)
                footLoc = pb.getLinkState(botId, linkIndex=3)[0][2]
                if botPos[2] < 0.55 or botPos[2] > 2 or footLoc > botPos[2]:
                    break
            pb.removeBody(botId)
        pb.stopStateLogging(loggingId = logID)
                    
    '''To get tarsa in line with tibia, angle should be (+) 133*np.pi/180
    to get foot in line with tarsa, angle should be (+)168*np.pi/180'''