" .py file to run DQN"

import numpy as np
import pandas as pd
import scipy.integrate as scp
import matplotlib.pyplot as plt
import numpy.random as rnd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
eps  = np.finfo(float).eps
import csv

from Experiment_online import Experiment as ExpTrain
from PolicyGradient import PGAgent
from Environment_Cont_x2_SIMO import Model_env


def EpochNoMean(data, bracket):
    data = np.stack(data)
    nrows = int(data.shape[0]/bracket)
    plot_prep_mean = np.zeros((int(nrows)))
    for f in range(0,nrows):
        x = data[f*bracket:f*bracket+ bracket-1]
        y = np.mean(x,0)
        plot_prep_mean[f] = y
    return plot_prep_mean

#plot of 1000 epoch mean throughout training
def Plotting(data, bracket, pNo_mean, key, xlabel):
    key = str(key)
    plt.figure(figsize =(20,16))
    plt.scatter(np.linspace(0, len(data), len(data)), data, label= 'Mean R over 1000 epochs')
    plt.xlabel(str(xlabel),  fontsize=28)
    plt.ylabel('Mean Return from Initial State', fontsize=28)
    #plt.title('Mean Reward over ' + str(bracket) + ' epochs with training')
    plt.tick_params(labelsize=24)
    plt.savefig('C:\\Users\\g41606mm\\Dropbox\\Projects\\RL\\PolicyGradient\\SystemDesign\\Accumulated_reward_' + str(pNo_mean) + '_' + str(key) + '_.png')
    return 

####################### -- Defining Parameters and Designing Experiement -- #########################

# Model definitions: parameters, steps, tf, x0  
      
p        = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}                 # model parameter definitions
steps_   = np.array([10])                                                                       # number of control interactions
tf       = 16.*24                                                 
x0       = np.array([0.5,150.0])                                                                # initial conditions of environment
                                                                                                # state space upper bound

# Agent definitions: Control UB, state_dim, discount factor                                     # number (range) of actions available to agent
disc1 = np.array([0.45])                                                                        # discount factor in back-allocation


# Experiment defintions: env, agent, controls, episodes
epochs              = 800                                       # number of training epochs
episodes            = 500                                       # number of episodes per epoch                                                                          
UB                  = [8, 2]                                    # upper bound on mean and std
BCs                 = [10, 0]                                   # control bounds

# miscellaneous algorithm hyperparameters
env_dict            = { 'X_sens': 100, 'N_sens': -1}        # defining objective function
agent_dict          = { 'obs_size': 2,
                        'n_traj': episodes,
                        'gamma': 0.99,
                        'hs_1': int(20),
                        'hs_2': int(15),
                        'output_size': int(1),
                        'U_UB': BCs[0],
                        'U_LB': BCs[1]}                        # constructing agent and indicating training


#run training 
env                         = Model_env(p, steps_, tf, x0, **env_dict)                          # calling environment
agent                       = PGAgent(steps_, **agent_dict)                                     # calling agent
experiment                  = ExpTrain(env, agent, episodes, epochs, disc1)                     # calling training experiment
reward_training, rv_m, d    = experiment.simulation()                                           # running training experiment
bracket                     = 100
reward_train_mean           = EpochNoMean(reward_training[:],bracket)
x_o = Plotting(reward_train_mean, bracket, UB ,'ruleallocation_', 'Training epochs (1e3)')
x_1 = Plotting(rv_m, bracket, UB ,'ruleallocation', 'Validation Run')

np.save("C:\\Users\\g41606mm\\Dropbox\\Projects\\RL\\PolicyGradient\\SystemDesign\\LSTM\\reward_training.py" , reward_training)
np.save("C:\\Users\\g41606mm\\Dropbox\\Projects\\RL\\PolicyGradient\\SystemDesign\\LSTM\\reward_validation_mean.py" , rv_m)

