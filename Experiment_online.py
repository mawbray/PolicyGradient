# Import 
import numpy as np
import scipy.integrate as scp
from tqdm import tqdm 
import numpy.random as rnd
eps  = np.finfo(float).eps
import h5py

################# --- Training Agent --- #####################
class Experiment(object):
    def __init__(self, env, agent, episodes, epochs, disc1):
      self.env , self.agent     = env, agent 
      self.episodes             = episodes
      self.disc1                = disc1
      self.epochs               = epochs
       
    def objective(self, terminal_state):
        obj_fun             = self.env.rewardfinal(terminal_state)
        return obj_fun

    def credit_assignment(self, s, transition):
        rwd             = self.env.reward(transition, self.disc1, s)
        return rwd
    
    def step(self, current_state, action, dt, s):

        prev_state      = current_state
        ode             = scp.ode(self.env.model)                           # define ode
        ode.set_integrator('lsoda', nsteps=3000)                            # define integrator
        ode.set_initial_value(current_state,dt)                             # set initial value
        ode.set_f_params(action)                                            # set control action
        current_state   = list(ode.integrate(ode.t + dt))                   # integrate system
        current_state   = self.correct_env((current_state))                      # correcting environment 
        transition      = [prev_state, current_state]                       # storing transition
        rwd             = self.credit_assignment(s, transition)             # reward allocation
        return current_state, rwd

    def simulation(self):
      # Simulation takes environment, imparts control action from e-greedy policy and simulates, observes next state to the end of the sequence and outputs reward
      # internal definitions
      self.correct_env      = self.env.correct_env
      dt, movements, x0     = self.env.dt, int(self.env.tf/float(self.env.dt)), self.env.x0
      model                 = self.env.model
      episodes              = self.episodes
      epochs                = self.epochs

      # compile state and control trajectories
      evaluation_period = 0.01
      nevals            = int(1/evaluation_period)
      rv_m              = []
      valid_eps         = 1000
      reward            = []

      for jj in tqdm(range(epochs)):
          xt              = []
          tt              = []
          c_hist          = []
          rwrd            = []
          

          for ei in (range(episodes)):
            # initialize simulation

            current_state   = x0
            xt.append(current_state)
            tt.append([float(0.)])
            
       
            # simulation of trajectory
            for s in range(movements):
                prev_state          = current_state
                action, logGP       = self.agent.act(current_state,s)                   # select control for this step from that possible
                c_hist.append(action)                                                   # storing control history for each epoch
                current_state, rwd  = self.step(current_state, action, dt, s)
                time                = (s+1)*dt
                tt.append(time)
                xt.append(current_state)                                                # add current state to history 
                rwrd.append(rwd)

                self.agent.Observe(prev_state, logGP, rwd)

            terminal_state      = current_state
            obj_fun             = self.objective(terminal_state)
            reward.append(obj_fun)
          
          
          self.agent.StoreEpisode(xt, tt, c_hist, rwrd, jj)
          self.agent.update_model()                                                     # weight update at the end of each epoch

          if jj % (evaluation_period * epochs) == 0:

              rv          = []
          
              for x in range(valid_eps):
              
                  current_state   = x0

                  for s in range(movements):

                        action, _       = self.agent.act(current_state, s)                 # select control for this step from that possible
                        ode             = scp.ode(self.env.model)                       # define ode
                        ode.set_integrator('lsoda', nsteps=3000)                        # define integrator
                        ode.set_initial_value(current_state,dt)                         # set initial value
                        ode.set_f_params(action)                                        # set control action
                        current_state = list(ode.integrate(ode.t + dt))                 # integrate system
                        current_state = self.correct_env(np.array(current_state))
                    
                  current_state   = current_state.reshape((2))
                  rv.append(self.objective(current_state))

              rv_m.append(np.stack(rv).mean())
    
      d     = self.agent.Learned()
      f     = h5py.File("validation_runs_store.hdf5",'w')
      grp   = f.create_group("AgentLC")
      dset  = grp.create_dataset("ValidRun", data = np.array(rv_m))
      dset[:] = np.array(rv_m)
      return reward, rv_m,  d 
  
