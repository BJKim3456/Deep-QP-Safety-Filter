import numpy as np
from numpy.random import rand
from math import exp

class bjkimenv:
    def __init__(self):
        self.action_dim = None
        self.state_dim = None
        self.action_high = None
        self.action_low = None
        self.name = None
    def GetActionInfo(self):
        return self.action_dim, self.action_high, self.action_low
    def GetEnvName(self):
        return self.name
    
class DoubleIntegrator(bjkimenv):
    def __init__(self, dt_=0.01, T_max = 20):
        super().__init__()
        self.dt_ = 0.001 
        self.rep_ = int(dt_/self.dt_)
        self.x = None
        self.v = None
        self.len = 0.1
        
        self.T_max_ = T_max
        self.cur_t = None
        
        self.action_dim = 1
        self.action_high = 1
        self.action_low = -1
        self.name = "DoubleIntegrator"
        
    def step(self, u):
        u = np.clip(u, self.action_low, self.action_high)
        for _ in range(self.rep_):
            ### RK4
            temp_v1 = self.v + 0.5*self.dt_*u
            temp_v2 = self.v + 0.5*self.dt_*u
            temp_v3 = self.v + self.dt_*u
            self.x = self.x + self.dt_/6*(self.v + 2*temp_v1 + 2*temp_v2 + temp_v3)
            self.v = self.v + self.dt_*u
        
        action_reward = exp(-(u/self.action_high)**2) - exp(-1) 
               
        diff_p = 1.5 - self.len + self.x 
        reward = exp( -diff_p@diff_p  ) + 0.1*action_reward
        
        constraint, info_constraint = self.getMinContraint()
        
        Trun = False
        Fail = False
        if (constraint < 0):
            Fail = True
        
        self.cur_t += self.dt_*self.rep_
        if (self.cur_t > self.T_max_):
            Trun = True
            
        return np.hstack((self.x, self.v)), reward, constraint, Fail, Trun, info_constraint
    
    def GetAnalyticValue(self): # only used for reset function
        x = self.x
        v = self.v
        if v >= 0:
            if (x>=0):
                value_ = 1.4 - x - 0.5*v**2
            elif(x<0 and v >= np.sqrt(-4*x)):
                value_ = 1.4 - x - 0.5*v**2
            else:
                value_ = 1.4 + x
        else:
            if (x<= 0):
                value_ = x - 0.5*v**2 + 1.4
            elif(x > 0 and v <= -np.sqrt(4*x)):
                value_ = x - 0.5*v**2 + 1.4
            else:
                value_ = 1.4 - x
        return value_
    def getMinContraint(self):
        violated_ = {'position': False}
        val_list = [1.5 - self.len - self.x, 1.5 - self.len + self.x]
        min_val = np.min(val_list)

        if val_list[0] < 0 or val_list[1] < 0:
            violated_['position'] = True
        
        return min_val, violated_

    def reset(self):
        v = -100
        while v < 0:
            self.x = (2*rand()-1)*2
            self.v = (2*rand()-1)*2
            v = self.GetAnalyticValue()
        self.cur_t = 0
        constraint_, _ = self.getMinContraint()
        return np.hstack((self.x, self.v)), constraint_
    
    