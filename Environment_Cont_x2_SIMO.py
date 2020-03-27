# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:34:05 2019
@author: Max Mowbray - University of Manchester, Manchester, United Kingdom
"""

import numpy as np
import torch 
eps  = np.finfo(float).eps

############ Defining Environment ##############

class Model_env: 
    
    # --- initializing model --- #
    def __init__(self, parameters, steps, tf, x0, **kwargs):
        
        # Object variable definitions
        self.parameters, self.steps = parameters, steps
        self.x0, self.dt, self.tf   = x0, tf/steps, tf                

        # objective function definitions
        self.args               = kwargs
        self.dRdX, self.dRdN    = self.args['X_sens'], self.args['N_sens']
        self.sensitivity        = [self.dRdX, self.dRdN]
        
        
    # --- dynamic model definition --- #    
    # model takes state and action of previous time step and integrates -- definition of ODE system at time, t
    def model(self, t, state, control):
        # internal definitions
        params = self.parameters
        FCn   = control
                
        # state vector
        Cx  = state[0]
        Cn  = state[1]
        
        # parameters
        u_m  = params['u_m']; K_N  = params['K_N'];
        u_d  = params['u_d']; Y_nx = params['Y_nx'];
        
        # algebraic equations
        
        # variable rate equations
        dev_Cx  = u_m * Cx * Cn/(Cn+K_N) - u_d*Cx
        dev_Cn  = - Y_nx * u_m * Cx * Cn/(Cn+K_N) + FCn
        
        return np.array([dev_Cx, dev_Cn],dtype='float64')
    
    def correct_env(self, state):
      
        for s in range(len(state)):
            if state[s] < 0:
                state[s] =0
            if state[s] == eps:
                state[s] = 0
        return state

    def reward(self, transition, disc1, s):
        
        # unpacking suitcase
        prev_state, current_state = transition
        
        time_to_term = self.steps - s
        y           = self.steps* 0.5 - s
        exp_        = np.exp(y)
        driver = []

        for prev, current, s in zip(prev_state, current_state, self.sensitivity):
            dSdt        = current - prev
            if s == 100:
                varderiv    = dSdt * s * exp_
            else:      
                varderiv    = dSdt * s * np.array([1])

            driver.append(varderiv)

        driver = np.stack(driver).sum()
        reward = (disc1**(time_to_term)) * (1/(1 + exp_)) * driver/10
        
        return reward

    def rewardfinal(self, terminal_state):
        driver =[]
        for state, s in zip(terminal_state, self.sensitivity):
            varderiv    = state * s
            driver.append(varderiv)

        reward = np.stack(driver).sum()

        return reward



        

