import numpy as np 
import pandas as pd
import torch
import timeit
from ANNClass import Net as PolicyNet
from ReplayMemory import ReplayMemory as RollOutMemory
from torch.distributions import Normal as norm
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import AffineTransform
import math
import h5py
eps  = np.finfo(float).eps


class Agent(object):  
        
    def __init__(self, movements):
        self.movements      = movements

    def act(self, state):
        raise NotImplementedErrorqs

class PGAgent(Agent):
    """Q-Learning agent with function approximation."""

    def __init__(self, movements, **kwargs):
        super(PGAgent, self).__init__(movements)

        # Definitions
        self.args           = kwargs
        # Agent Topology
        self.obs_size       = self.args['obs_size']
        self.inputZ         = int(self.obs_size + 3)

        # code regularisation
        self.dtype          = torch.float
        self.use_cuda       = torch.cuda.is_available()
        self.device         = torch.device("cuda:0" if self.use_cuda else "cpu")
        torch.cuda.empty_cache() 


        # miscellaneous algorithm hyperparameters
        self.movements          = movements
        self.n_traj             = self.args['n_traj']                      # n trajectories in an epoch
        self.action_rolls       = int(self.movements * self.n_traj)     # N transitions in an epoch
        self.gamma              = self.args['gamma']                       # discount future return

        # defining agent topology and optimisation routine
        net_kwargs      = {'input_size': self.inputZ, 
                           'hs_1': self.args['hs_1'],
                           'hs_2': self.args['hs_2'], 
                           'output_size': self.args['output_size']}

        
        self.policy_network     = PolicyNet(**net_kwargs).to(self.device)
        self.optimizer          = torch.optim.RMSprop(self.policy_network.parameters(), lr=kwargs.get('learning_rate', 5e-2))                                   # Stochastic Gradient Descent
        self.current_loss       = .0                                                                                    

        # storing and standardising states
        self.memory             = RollOutMemory(self.obs_size, self.action_rolls)
        self.ep_count           = 0.


        # initialising h5py file for dataset storage and collation
        self.f  = h5py.File("trajectory_transition_store.hdf5",'w')



    def hiDprojection(self, state, T= None):

        (X_max, X_min), (N_max, N_min) = (2, 0), (1000, 0)              # normalisation
        
        X, N = state[0], state[1]
        X, N = (X - X_min)/(X_max - X_min), (N - N_min)/(N_max - N_min)
        output = torch.zeros((self.inputZ), dtype = torch.float64)
        output[0], output[1], output[2], output[3], output[4] = X, N, X * N, X ** 2, N ** 2
        
        return output
    

    def SampleAction(self, mean, std):
        # mean and ln_var are predicted by the neural network, this function
        mu      = mean
        sig     = std * 0.3                             # constraining the standard deviation to at maximum 0.3mu
        u_range = self.args['U_UB'] - self.args['U_LB']

        GPol    = norm(mu, sig)                         # defining gaussian distribution with mean and std as parameterised
        scale   = AffineTransform(self.args['U_LB'], u_range)
        GPol    = TransformedDistribution(GPol, scale)

        action  = GPol.sample()                         # drawing randomly from normal distribution 
        assert len(action) == 1
        logGP   = GPol.log_prob(action)                  # calculating log probability of action taken

        return action.cpu(), logGP

    def StoreEpisode(self,states, time, actions, rwd_ass, epoch):
        # create datafile for each epoch
        
        grp     = self.f.create_group(f"trajectories_epoch_{epoch}")
        subgrp1 = grp.create_group('states')
        ds1     = subgrp1.create_dataset('state_traj', data=np.array(states))
        subgrp2 = grp.create_group('actions')
        ds2     = subgrp2.create_dataset('action_traj', data=np.array(actions))
        subgrp3 = grp.create_group('rwd_ass')
        ds3     = subgrp3.create_dataset('rwd_traj', data=np.array(rwd_ass))
        subgrp5 = grp.create_group('time')
        ds5     = subgrp5.create_dataset('time_traj', data=np.array(time))


    def act(self, state,s):
              
        self.policy_network.eval()
        state           = self.hiDprojection(state,1)
        input           = state.reshape(1,state.shape[0])
        input           = input.clone().detach().to(self.device)
        P               = self.policy_network(input.float())
        #P              = P.detach().numpy().cpu().squeeze()
        mu, std         = P[0], P[1]
        action, logGP   = self.SampleAction(mu, std) 

        if action < self.args['U_LB']:
            action = self.args['U_LB']
        if action > self.args['U_UB']:
            action = self.args['U_UB']
        
        #print(logGP)
        

        return action, logGP     
    
    def Observe(self, state, logGP, reward):
        self.memory.observe(state, reward, logGP)
        return 

    def update_model(self):
        self.ep_count += 1
        #self.memory.normalised_it()
        (logP, reward) = self.memory.sample_minibatch(self.action_rolls)
    
        # Compute Policy Gradient approx

        discounted_rewards = []
        LogPG = []
        n_traj      = int(self.action_rolls/self.movements)
        J_perf      = 0.

        """                                            
        Calculating return from each trajectory
        and summing log probabilities of each action taken
        in each trajectory
                                                        """

        for ii in range(self.n_traj):
            Gt = 0.
            pw      = 0.
            logprob = 0.
            index   = int(ii * self.movements)
            index_1 = int(index + self.movements)
            for g in range(int(self.movements)):
                Gt      += self.gamma**pw * reward[index+g]             # calculating return from trajectory 
                logprob += logP[index + g]                              # summing log probability of actions in trajectory
                pw += 1
            
            
            LogPG.append(logprob)
            discounted_rewards.append(Gt)

        

        # REINFORCE with Baseline
        LogPG              = torch.tensor(LogPG, dtype = torch.float64)
        discounted_rewards = torch.tensor(discounted_rewards, dtype = torch.float64)
        baseline           = torch.mean(discounted_rewards, dim=0)
        std                = torch.std(discounted_rewards) 
        discounted_rewards = (discounted_rewards - baseline)     
        

        assert LogPG.shape[0] == discounted_rewards.shape[0],    "tensors must be the same shape"
        
        # Constructing PG associated with each trajectory
        policy_gradient = []
        for logprob, Gt in zip(LogPG, discounted_rewards):
            policy_gradient.append(-logprob * Gt)

        policy_gradient = torch.stack(policy_gradient).mean()        # taking the mean of the policy gradient over all trajectories gathered
        policy_gradient = policy_gradient.clone().detach().requires_grad_(True)
        
        print(policy_gradient)
        # perform model update
        self.policy_network.train()
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() 

        return policy_gradient.item()
    
    def Learned(self):
        return self.policy_network



 
    


