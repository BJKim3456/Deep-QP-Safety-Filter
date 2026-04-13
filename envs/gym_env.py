import gymnasium as gym
import numpy as np
from pathlib import Path

BASE_TIMESTEP = 0.001
ASSET_DIR = Path(__file__).resolve().parent / "assets"

def get_frame_skip(dt_: float) -> int:
    frame_skip = int(round(dt_ / BASE_TIMESTEP))
    if not np.isclose(frame_skip * BASE_TIMESTEP, dt_):
        raise ValueError(
            f"dt_={dt_} must be an integer multiple of {BASE_TIMESTEP}"
        )
    return frame_skip

def get_xml_path(filename: str) -> str:
    return str(ASSET_DIR / filename)


class bjkimenv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_dim = None
        self.action_high = None
        self.action_low = None
        self.name = None
        self.env_ = None
    def GetActionInfo(self):
        return self.action_dim, self.action_high, self.action_low
    def GetEnvName(self):
        return self.name
    def SetEnvSeed(self, seed):
        self.env_.reset(seed = seed)
        self.env_.action_space.seed(seed)
        self.env_.observation_space.seed(seed)
    
class InvertedPendulum(bjkimenv):
    def __init__(self, dt_=0.01, T_max = 20, render_ = None, custom_reward_bonus = False, seed = 0):
        super().__init__()
        frame_skip = get_frame_skip(dt_)
        xml_file = get_xml_path("inverted_pendulum_dt001.xml")

        if render_ is not None:
            self.env_ = gym.make(
                "InvertedPendulum-v5",
                max_episode_steps=int(T_max / dt_),
                render_mode=render_,
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        else:
            self.env_ = gym.make(
                "InvertedPendulum-v5",
                max_episode_steps=int(T_max / dt_),
                frame_skip=frame_skip,
                xml_file=xml_file,
            )

        self.name = "1D_Inverted_Pendulum"
        self.action_dim = self.env_.action_space.shape[0]
        self.action_high = self.env_.action_space.high
        self.action_low = self.env_.action_space.low
        self.state = None
        self.custom_reward_bonus_ = custom_reward_bonus
        self.SetEnvSeed(seed = seed)
        
    def step(self, u):
        self.state, reward, term, truc, _ = self.env_.step(u)
        constraint,violated_info = self.getMinContraint()
        if (constraint < 0):
            term = True
        if (self.custom_reward_bonus_):
            reward += np.exp(-(1-self.state[0])**2)
        return self.state, reward, constraint, term, truc, violated_info
        
    def reset(self):
        self.state, _ = self.env_.reset()
        constraint,_ = self.getMinContraint() 
        return self.state, constraint
    
    def getMinContraint(self):
        x = self.state[0]
        angle = self.state[1]
        
        min_val_list = []
        min_val_list.append(0.2 - angle)
        min_val_list.append(0.2 + angle)
        min_val_list.append(1 + x)
        min_val_list.append(1 - x)
        min_val = min(min_val_list)
        violated_ = {'tip': False, 'x_pos': False}
        if min_val_list[0] < 0 or min_val_list[1] < 0 :
            violated_['tip'] = True
        if min_val_list[2] < 0 or min_val_list[3] < 0:
            violated_['x_pos'] = True
        
        return min_val, violated_
    
class InvertedDoublePendulum(bjkimenv):
    def __init__(self, dt_=0.01, T_max = 20, render_ = None, custom_reward = False, seed = 0):
        super().__init__()
        frame_skip = get_frame_skip(dt_)
        xml_file = get_xml_path("inverted_double_pendulum_dt001.xml")

        if render_ is not None:
            self.env_ = gym.make(
                "InvertedDoublePendulum-v5",
                max_episode_steps=int(T_max / dt_),
                render_mode=render_,
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        else:
            self.env_ = gym.make(
                "InvertedDoublePendulum-v5",
                max_episode_steps=int(T_max / dt_),
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        self.name = "2D_Inverted_Pendulum"
        self.action_dim = self.env_.action_space.shape[0]
        self.action_high = self.env_.action_space.high
        self.action_low = self.env_.action_space.low
        self.custom_reward = custom_reward
        self.state = None
        self.SetEnvSeed(seed = seed)
        
    def step(self, u):
        self.state, reward, term, truc, _ = self.env_.step(u)
        constraint, violated_info = self.getMinContraint()
        if ( constraint  < 0):
            term = True
        if (self.custom_reward):
            reward = abs(self.state[5])
        return self.state, reward, constraint, term, truc, violated_info
        
    def reset(self):
        self.state, _ = self.env_.reset()
        constraint, _ = self.getMinContraint()
        return self.state, constraint
    
    def getMinContraint(self):
        _, _, y = self.env_.unwrapped.data.site_xpos[0]

        min_val_list = [(y - 1), (self.state[0]+0.95), (0.95-self.state[0])]
        violated_ = {'tip': False, 'x-pos': False}

        if min_val_list[0] < 0 :
            violated_['tip'] = True
        if min_val_list[1] < 0 or min_val_list[2] < 0:
            violated_['x_pos'] = True
        
        return min(min_val_list), violated_
    
class InvertedDoublePendulumPositionBonus(bjkimenv):
    def __init__(self, dt_=0.01, T_max = 20, render_ = None, custom_reward = False, seed = 0):
        super().__init__()
        
        frame_skip = get_frame_skip(dt_)
        xml_file = get_xml_path("inverted_double_pendulum_dt001.xml")

        if render_ is not None:
            self.env_ = gym.make(
                "InvertedDoublePendulum-v5",
                max_episode_steps=int(T_max / dt_),
                render_mode=render_,
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        else:
            self.env_ = gym.make(
                "InvertedDoublePendulum-v5",
                max_episode_steps=int(T_max / dt_),
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        self.name = "InvertedDoublePendulumPositionBonus"
        self.action_dim = self.env_.action_space.shape[0]
        self.action_high = self.env_.action_space.high
        self.action_low = self.env_.action_space.low
        self.custom_reward = custom_reward
        self.state = None
        self.SetEnvSeed(seed = seed)
        
    def step(self, u):
        self.state, _, term, truc, _ = self.env_.step(u)
        constraint, violated_info = self.getMinContraint()
        if ( constraint  < 0):
            term = True
        reward = abs(self.state[0])  if not term else -1
        return self.state, reward, constraint, term, truc, violated_info
        
    def reset(self):
        self.state, _ = self.env_.reset()
        constraint, _ = self.getMinContraint()
        return self.state, constraint
    
    def getMinContraint(self):
        _, _, y = self.env_.unwrapped.data.site_xpos[0]

        min_val_list = [(y - 1), (self.state[0]+0.95), (0.95-self.state[0])]
        violated_ = {'tip': False, 'x-pos': False}

        if min_val_list[0] < 0 :
            violated_['tip'] = True
        if min_val_list[1] < 0 or min_val_list[2] < 0:
            violated_['x_pos'] = True
        
        return min(min_val_list), violated_
    
class InvertedDoubleMovingBonus(bjkimenv):
    def __init__(self, dt_=0.01, T_max = 20, render_ = None, custom_reward = False, seed = 0):
        super().__init__()
        
        frame_skip = get_frame_skip(dt_)
        xml_file = get_xml_path("inverted_double_pendulum_dt001.xml")

        if render_ is not None:
            self.env_ = gym.make(
                "InvertedDoublePendulum-v5",
                max_episode_steps=int(T_max / dt_),
                render_mode=render_,
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        else:
            self.env_ = gym.make(
                "InvertedDoublePendulum-v5",
                max_episode_steps=int(T_max / dt_),
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        self.name = "InvertedDoubleMovingBonus"
        self.action_dim = self.env_.action_space.shape[0]
        self.action_high = self.env_.action_space.high
        self.action_low = self.env_.action_space.low
        self.custom_reward = custom_reward
        self.state = None
        self.SetEnvSeed(seed = seed)
        
    def step(self, u):
        self.state, _, term, truc, _ = self.env_.step(u)
        constraint, violated_info = self.getMinContraint()
        if ( constraint  < 0):
            term = True
        reward = abs(self.state[5]) if not term else -1
        return self.state, reward, constraint, term, truc, violated_info
        
    def reset(self):
        self.state, _ = self.env_.reset()
        constraint, _ = self.getMinContraint()
        return self.state, constraint
    
    def getMinContraint(self):
        _, _, y = self.env_.unwrapped.data.site_xpos[0]

        min_val_list = [(y - 1), (self.state[0]+0.95), (0.95-self.state[0])]
        violated_ = {'tip': False, 'x-pos': False}

        if min_val_list[0] < 0 :
            violated_['tip'] = True
        if min_val_list[1] < 0 or min_val_list[2] < 0:
            violated_['x_pos'] = True
        
        return min(min_val_list), violated_
        
class Hopper(bjkimenv):
    def __init__(self, dt_=0.01, T_max = 20, render_ = None, seed = 0):
        super().__init__()
        frame_skip = get_frame_skip(dt_)
        xml_file = get_xml_path("hopper_dt001.xml")

        if render_ is not None:
            self.env_ = gym.make(
                "Hopper-v5",
                max_episode_steps=int(T_max / dt_),
                render_mode=render_,
                frame_skip=frame_skip,
                xml_file=xml_file,
            )
        else:
            self.env_ = gym.make(
                "Hopper-v5",
                max_episode_steps=int(T_max / dt_),
                frame_skip=frame_skip,
                xml_file=xml_file,
            )

        self.name = "Hopper"
        self.action_dim = self.env_.action_space.shape[0]
        self.action_high = self.env_.action_space.high
        self.action_low = self.env_.action_space.low
        self.state = None
        
        self.min_z, self.max_z = self.env_.unwrapped._healthy_z_range
        self.min_angle, self.max_angle = self.env_.unwrapped._healthy_angle_range
        self.SetEnvSeed(seed = seed)

    def step(self, u):
        self.state, reward, term, truc, info_env = self.env_.step(u)
        constraint, info_constraint = self.getMinContraint()
        return self.state, reward, constraint, term, truc, info_constraint
        
    def reset(self):
        self.state, _ = self.env_.reset()
        constraint,_ = self.getMinContraint() 
        return self.state, constraint
    
    def getMinContraint(self):
        z = self.state[0]
        angle = self.state[1]
        
        violated_ = {'height': False, 'angle': False}
        
        min_val_list = [-self.min_z + z, self.max_angle - angle, -self.min_angle + angle]
        
        min_val = np.min(min_val_list)
        if min_val_list[0] < 0 :
            violated_['height'] = True
        if min_val_list[1] < 0 or min_val_list[2] < 0:
            violated_['angle'] = True
        return min_val, violated_
    