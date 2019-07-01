# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:46:01 2019

@author: Sam
"""

import vrep # access all the VREP elements
import numpy as np

class VREPit():
    def __init__(self):
        self.handles = []
        self.clientID = 0
        
        
        self.Connect()
        
        
        
    def Connect(self):
        err = 0
        
        vrep.simxFinish(-1) # just in case, close all opened connections
        clientID=vrep.simxStart("127.0.0.1",19999,True,True,5000,5) # start a connection
        self.clientID = clientID
        if clientID!=-1:
            print ("Connected to remote API server")
        else:
            print("Not connected to remote API server")
        
        
        err_code,base_h = vrep.simxGetObjectHandle(clientID, "base_link_visual", vrep.simx_opmode_blocking); err+=err_code
        
        err_code,fem_l_h= vrep.simxGetObjectHandle(clientID,"Hip_femur_L", vrep.simx_opmode_blocking); err+=err_code
        err_code,tib_l_h = vrep.simxGetObjectHandle(clientID,"Femur_tibia_L", vrep.simx_opmode_blocking); err+=err_code
        err_code,tar_l_h = vrep.simxGetObjectHandle(clientID,"Tibia_tarsa_L", vrep.simx_opmode_blocking); err+=err_code
        err_code,foo_l_h = vrep.simxGetObjectHandle(clientID,"Tarsa_foot_L", vrep.simx_opmode_blocking); err+=err_code
        
        
        err_code,fem_r_h = vrep.simxGetObjectHandle(clientID,"Hips_femur_R", vrep.simx_opmode_blocking); err+=err_code
        err_code,tib_r_h = vrep.simxGetObjectHandle(clientID,"Femur_tibia_R", vrep.simx_opmode_blocking); err+=err_code
        err_code,tar_r_h = vrep.simxGetObjectHandle(clientID,"Tibia_tarsa_R", vrep.simx_opmode_blocking); err+=err_code
        err_code,foo_r_h = vrep.simxGetObjectHandle(clientID,"Tarsa_foot_R", vrep.simx_opmode_blocking); err+=err_code
    
        self.handles = [base_h, fem_l_h, tib_l_h, tar_l_h, foo_l_h, fem_r_h, tib_r_h, tar_r_h, foo_r_h]
        return err
        
    
    def getAngles(self):
        clientID = self.clientID
        h = self.handles
        
        angles = []
        for i in range(8):
            err_code, pos = vrep.simxGetJointPosition(clientID, h[i+1], vrep.simx_opmode_blocking)
            angles.append((pos*180/np.pi)%360)
          
        return angles
    
    def setAngles(self, angles = [0,0,0,0,0,0,0,0]):
        clientID = self.clientID
        h = self.handles
        
        vrep.simxPauseSimulation(clientID,vrep.simx_opmode_oneshot)
        for i in range(8):
            err_code = vrep.simxSetJointPosition(clientID, h[i+1], angles[i]*np.pi/180, vrep.simx_opmode_oneshot)
        vrep.simxPauseSimulation(clientID,False); vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        
        return err_code
        
    def getPosOrn(self, handlenum=-1):
        clientID = self.clientID
        
        if handlenum == -1:
            handlenum = self.handles[0]
            
            
        err_code, pos = vrep.simxGetObjectPosition(clientID, handlenum, -1, vrep.simx_opmode_blocking)
        err_code, orn = vrep.simxGetObjectOrientation(clientID, handlenum, -1, vrep.simx_opmode_blocking)
        
        if err_code == 0:
            return [pos, orn]
        else:
            return err_code
