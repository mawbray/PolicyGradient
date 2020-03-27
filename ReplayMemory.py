import numpy as np
import torch

class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = int(observation_size)
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                    'obs'      : np.zeros((self.max_size * 1 * self.observation_size),
                                       dtype= np.float64).reshape(self.max_size, self.observation_size),
                    'reward'   : np.zeros((self.max_size, 1), dtype = np.float64).reshape(self.max_size, 1),
                    'logP'     : np.zeros((self.max_size, 1), dtype = np.float64).reshape(self.max_size, 1)}

        self.dtype          = torch.float64
        self.use_cuda       = torch.cuda.is_available()
        self.device         = torch.device("cuda:0" if self.use_cuda else "cpu")

    def observe(self, state, reward, logGP):
        index = self.num_observed % self.max_size
     
        self.samples['obs'][index, :]        = state
        self.samples['logP'][index,:]       = logGP.cpu().detach().numpy()
        self.samples['reward'][index, :]    = reward


        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.linspace(0,minibatch_size-1, minibatch_size, dtype = np.int64)

        logP   = torch.tensor((self.samples['logP'][sampled_indices].reshape(minibatch_size,1)), dtype = self.dtype).cpu()
        r      = torch.tensor((self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))),dtype = self.dtype).cpu()
      


        return (logP, r)

    def normalised_it(self):
        
        s       = self.samples['obs'][:, :]

        s_mean  = np.mean(s, axis = 0 , dtype = np.float64)
        s_std   = np.std(s, axis = 0 , dtype = np.float64)

        self.samples['obs'][:, :]   = (s - s_mean)/s_std
        return

    def normalisation(self):
        s       = self.samples['obs'][:, :]
        s_mean  = np.mean(s, axis = 0 , dtype = np.float64)
        s_std   = np.std(s, axis = 0 , dtype = np.float64)
        return s_mean, s_std


        





