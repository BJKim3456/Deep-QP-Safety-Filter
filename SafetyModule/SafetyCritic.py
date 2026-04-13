import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
#################################################
########## for QP-based safety filter ###########
#################################################

from ReplayBuffer.ReplayBuffer import ReplayMemory

TERMINAL_ = -0.05
VALUE_SCALE_UP = 1.01 
OFFSET = 0.05 

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # relu init for elu activations
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, num_hidden = 2 , dtype = torch.float32):
        super(MLPWithLayerNorm, self).__init__()
        self.input_layer_ = nn.Linear(input_dim, hidden_dim, dtype=dtype).to(device)
        self.input_layer_norm_ = nn.LayerNorm(hidden_dim, dtype=dtype).to(device)
        self.activ_input_ = nn.ELU().to(device)
         
        self.hidden_layers_ = nn.ModuleList()
        self.activ_hiddens_ = nn.ModuleList()
        self.layer_norms_hiddens_ = nn.ModuleList()
        self.num_hidden = num_hidden
        for _ in range(num_hidden):
            self.hidden_layers_.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype).to(device))
            self.activ_hiddens_.append(nn.ELU().to(device))
            self.layer_norms_hiddens_.append(nn.LayerNorm(hidden_dim, dtype=dtype).to(device))
            
        self.output_layer_ = nn.Linear(hidden_dim, output_dim, dtype=dtype).to(device)
            
        self.apply(weights_init_)
        
    def forward(self, x):
        x = self.activ_input_(self.input_layer_norm_(self.input_layer_(x)))
        for i in range(self.num_hidden):
            x = self.activ_hiddens_[i](self.layer_norms_hiddens_[i](self.hidden_layers_[i](x)))
        return self.output_layer_(x)
    
class ValueFtn(nn.Module):
    def __init__(self, state_dim, hidden_dim, device, num_hidden = 2, dtype=torch.float32):
        super(ValueFtn, self).__init__()
        
        self.NN_1 = MLPWithLayerNorm(state_dim, hidden_dim, 1, device, num_hidden = num_hidden, dtype=dtype)
        self.NN_2 = MLPWithLayerNorm(state_dim, hidden_dim, 1, device, num_hidden = num_hidden, dtype=dtype)
    
        self.output_activation_ = nn.Softplus().to(device)

    def GetValue(self, constraints, states, clip = True) -> tuple[torch.FloatTensor,torch.FloatTensor]:
        return_1_  = constraints - ( self.output_activation_( self.NN_1(states) ) - OFFSET)
        return_2_  = constraints - ( self.output_activation_( self.NN_2(states) ) - OFFSET)
        return (return_1_, return_2_) if not clip else (torch.clamp(torch.clamp(return_1_, max=constraints), min=TERMINAL_), torch.clamp(torch.clamp(return_2_, max=constraints), min=TERMINAL_))
        
class ScalarDeviation(nn.Module):
    def __init__(self, state_dim, hidden_dim, device, num_hidden = 2 , dtype=torch.float32):
        super(ScalarDeviation, self).__init__()
        self.NN_ = MLPWithLayerNorm(state_dim, hidden_dim, 1, device, num_hidden = num_hidden, dtype=dtype)
        self.output_activation_ = nn.Softplus().to(device)
    def forward(self, x):
        return self.output_activation_( self.NN_(x) ) - OFFSET
    
    def GetValue(self, lb, states, clip = True) -> torch.FloatTensor:
        scalar_ = self.forward(states) if not clip else torch.clamp(self.forward(states), min=0.0)
        return scalar_ - lb
    
class SafeValues(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device, num_hidden = 2, dtype = torch.float32):
        super(SafeValues, self).__init__()
        self.device = device
        self.safe_Value = ValueFtn(state_dim, hidden_dim, self.device, num_hidden=num_hidden, dtype=dtype)
        self.safe_q_scalar = ScalarDeviation(state_dim, hidden_dim, self.device, num_hidden=num_hidden, dtype=dtype)
        self.safe_q_coeff = MLPWithLayerNorm(state_dim, hidden_dim, action_dim, self.device, num_hidden=num_hidden, dtype=dtype)
        
class SafetyCritic():
    def __init__(
                 self,
                 state_dim, action_dim,
                 args:dict, eval_ = False
                 ):
        
        self.device_ = args['cuda']
        self.dtype_ = args['dtype']
        self.SafeValues_ = SafeValues(state_dim, args['hidden_size'], action_dim, self.device_, num_hidden=args['hidden_layers'], dtype=self.dtype_)
        self.TargetSafeValues_ = SafeValues(state_dim, args['hidden_size'], action_dim, self.device_, num_hidden=args['hidden_layers'], dtype=self.dtype_)
        self.TargetUpdateHelper(self.SafeValues_, self.TargetSafeValues_, 1)
        
        self.safety_lambda = np.maximum(args['lambda_init'], args['lambda_final'])
        self.c_max_ = 1e-3
        if not eval_:
            self.batch_size = args['batch_size']
            self.dt_ = args['dt']
            self.tau_safe_ = args['tau']
            self.action_dim_ = action_dim
            self.clip_grad_ = args['clip_grad']
            self.update_count_ = args['update_count']
            self.warmup_steps_ = args['warmup_steps']
            
            self.safety_lambda = self.safety_lambda if not args['with_warm_up'] else 0.1/args['dt']
            self.safety_lambda_start_to_decrease_ = args['lambda_start_to_decrease']
            self.safety_lambda_init_ = args['lambda_init']
            self.safety_lambda_decrease_rate_ = args['lambda_increasing_interval']/10
            self.safety_lambda_decrease_interval_ = args['lambda_increasing_interval']
            self.safety_lambda_goal_ = args['lambda_final']
            
            self.warm_up_ = args['with_warm_up']
            self.lr_value_max_ = args['lr_safe_value']
            self.lr_coeff_max_ = args['lr_safe_q_coeff']
            self.lr_scalar_max_ = args['lr_safe_q_scalar']
            self.lr_value_min_ = args['lr_safe_value_goal']
            self.lr_coeff_min_ = args['lr_safe_q_coeff_goal']
            self.lr_scalar_min_ = args['lr_safe_q_scalar_goal']
            self.lr_decrease_interval_ = args['lr_decreasing_interval']
            self.lr_value_decrease_ = (args['lr_safe_value_goal'] - args['lr_safe_value'])/args['lr_decreasing_interval']
            self.lr_coeff_decrease_ = (args['lr_safe_q_coeff_goal'] - args['lr_safe_q_coeff'])/args['lr_decreasing_interval']
            self.lr_scalar_decrease_ = (args['lr_safe_q_scalar_goal'] - args['lr_safe_q_scalar'])/args['lr_decreasing_interval']
            self.replay_memory = ReplayMemory(args['replay_size'])
            self.use_offline_data_ = False

            self.safe_q_optimizer_ = torch.optim.Adam([
                    {'params': self.SafeValues_.safe_q_coeff.parameters(), 'lr': args['lr_safe_q_coeff']},
                    {'params': self.SafeValues_.safe_q_scalar.parameters(), 'lr': args['lr_safe_q_scalar']},
                ])
            self.safe_value_optimizer_ = torch.optim.Adam(self.SafeValues_.safe_Value.parameters(), args['lr_safe_value'])
            
    def GetValues(self, state: torch.FloatTensor, current_constraint_value: float, UseTarget: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if current_constraint_value > self.c_max_:
            self.c_max_ = current_constraint_value
        with torch.no_grad():
            norm_c_ = current_constraint_value/self.c_max_
            networks_ = self.TargetSafeValues_ if UseTarget else self.SafeValues_

            v1, v2 = networks_.safe_Value.GetValue(norm_c_, state)
            v_ = torch.min(v1, v2)
            coeff = networks_.safe_q_coeff(state)
            scalar = networks_.safe_q_scalar.GetValue(self.safety_lambda * (norm_c_ - v_), state) - torch.sum(coeff.abs()) 
            
        return v_.cpu().numpy(), coeff.cpu().numpy(), scalar.cpu().numpy()
    
    def safe_critic_update_warmup(self,state_batch,action_batch,constraint_batch,next_constraint_batch):
        v_pi_1, v_pi_2 = self.SafeValues_.safe_Value.GetValue(constraint_batch, state_batch, clip=False)
        coeff = self.SafeValues_.safe_q_coeff(state_batch)
        
        next_constraint_batch = torch.clamp(next_constraint_batch, min=TERMINAL_) 

        scalar = self.SafeValues_.safe_q_scalar.GetValue(0.0, state_batch, clip=False) 

        dvdt_coeff_term_ = torch.sum(coeff*action_batch - coeff.abs(), dim=1, keepdim=True)

        safe_value_loss = F.mse_loss(v_pi_1, constraint_batch)/self.dt_ + F.mse_loss(v_pi_2, constraint_batch)/self.dt_
        self.safe_value_optimizer_.zero_grad()
        safe_value_loss.backward()   
        nn.utils.clip_grad_norm_(self.SafeValues_.safe_Value.parameters(), self.clip_grad_)
        self.safe_value_optimizer_.step()
        
        rhs_dvdt = (next_constraint_batch - constraint_batch)/self.dt_
        dvdt_coeff = dvdt_coeff_term_ + torch.clamp(scalar.detach().clone(), min=0.0)
        rhs_scalar = rhs_dvdt - dvdt_coeff_term_.detach().clone()
        rhs_scalar = torch.clamp(rhs_scalar, min=-OFFSET)
        safe_small_q_coeff_loss = F.mse_loss(dvdt_coeff, rhs_dvdt)*self.dt_
        safe_small_q_scalar_loss = F.mse_loss(scalar, rhs_scalar)*self.dt_ # 
        safe_small_q_loss = safe_small_q_coeff_loss + safe_small_q_scalar_loss
        
        self.safe_q_optimizer_.zero_grad()
        safe_small_q_loss.backward()
        nn.utils.clip_grad_norm_(self.SafeValues_.safe_q_coeff.parameters(), self.clip_grad_)
        nn.utils.clip_grad_norm_(self.SafeValues_.safe_q_scalar.parameters(), self.clip_grad_)
        self.safe_q_optimizer_.step()

        with torch.no_grad():
            min_v = v_pi_1 
            return safe_value_loss.item(), safe_small_q_loss.item(), torch.min(constraint_batch - min_v, scalar + self.safety_lambda*(constraint_batch - min_v)).mean().item()
    
    def safety_critic_update(self,state_batch,action_batch,constraint_batch,next_state_batch,next_constraint_batch):
        v_pi_1, v_pi_2 = self.SafeValues_.safe_Value.GetValue(constraint_batch, state_batch, clip=False)
        coeff = self.SafeValues_.safe_q_coeff(state_batch)
        
        lambda_dt_ = self.safety_lambda * self.dt_
        discount_ = np.exp(-lambda_dt_)
        next_constraint_batch = torch.clamp(next_constraint_batch, min=TERMINAL_)

        with torch.no_grad():
            v_pi_next_state_for_q, v_pi_next_state_2 = self.TargetSafeValues_.safe_Value.GetValue(next_constraint_batch, next_state_batch)
            v_pi_next_state = torch.min(v_pi_next_state_for_q, v_pi_next_state_2) 
            
            v_pi_for_q, _ = self.TargetSafeValues_.safe_Value.GetValue(constraint_batch, state_batch) 
            
            target_coeff = self.TargetSafeValues_.safe_q_coeff(state_batch)
            
            cVgap_ = constraint_batch - v_pi_for_q 
            scalar_lb_ = self.safety_lambda*cVgap_
            
            next_cVgap_ = next_constraint_batch - v_pi_next_state
            next_scalar_lb_ = self.safety_lambda*next_cVgap_

            next_close_enough = (next_cVgap_<=0.0)*1; next_close_enough[next_constraint_batch <= TERMINAL_] = 0
            target_next_scalar = ( 
                next_close_enough*self.TargetSafeValues_.safe_q_scalar.GetValue(next_scalar_lb_, next_state_batch) + (1-next_close_enough)*(-next_scalar_lb_) 
                )
            
            target_next_v = torch.clamp(torch.clamp( torch.min(next_constraint_batch, discount_*(v_pi_next_state + self.dt_ * target_next_scalar) + lambda_dt_*next_constraint_batch), min=TERMINAL_ ), max=next_constraint_batch)
        with torch.no_grad():
            target_coeff_term_ = torch.sum(target_coeff*action_batch - target_coeff.abs(), dim=1, keepdim=True)
            target_scalar = self.TargetSafeValues_.safe_q_scalar.GetValue(scalar_lb_, state_batch) 

            integral_exp_c_ = lambda_dt_*constraint_batch 

            rhs_dvdt = ( target_next_v - v_pi_for_q ) / self.dt_
            target_dvdt_plus_lambda_c_minus_v = self.dt_*(target_coeff_term_ + target_scalar - self.safety_lambda*v_pi_for_q) + integral_exp_c_

            q_part_for_v = torch.clamp(target_dvdt_plus_lambda_c_minus_v, max=0.0)
            
        rhs_value = torch.min(constraint_batch, discount_*v_pi_next_state + integral_exp_c_) - q_part_for_v

        safe_value_loss = F.mse_loss(v_pi_1, rhs_value)/self.dt_ + F.mse_loss(v_pi_2, rhs_value)/self.dt_ # F.mse_loss(v_pi_1/self.dt_, rhs_value/self.dt_) + F.mse_loss(v_pi_2/self.dt_, rhs_value/self.dt_) # 
        self.safe_value_optimizer_.zero_grad()
        safe_value_loss.backward()   
        nn.utils.clip_grad_norm_(self.SafeValues_.safe_Value.parameters(), self.clip_grad_)
        self.safe_value_optimizer_.step()
        
        scalar = self.SafeValues_.safe_q_scalar.GetValue(scalar_lb_, state_batch, clip=False) 

        dvdt_coeff_term_ = torch.sum(coeff*action_batch - coeff.abs(), dim=1, keepdim=True) # 
        
        rhs_scalar = rhs_dvdt - target_coeff_term_
        dvdt_coeff = dvdt_coeff_term_ + target_scalar 
        safe_small_q_coeff_loss = F.mse_loss(dvdt_coeff, rhs_dvdt)*self.dt_ 
        safe_small_q_scalar_loss = F.mse_loss(scalar, rhs_scalar)*self.dt_ 
        safe_small_q_loss = safe_small_q_coeff_loss + safe_small_q_scalar_loss
        
        self.safe_q_optimizer_.zero_grad()
        safe_small_q_loss.backward()
        nn.utils.clip_grad_norm_(self.SafeValues_.safe_q_coeff.parameters(), self.clip_grad_)
        nn.utils.clip_grad_norm_(self.SafeValues_.safe_q_scalar.parameters(), self.clip_grad_)
        self.safe_q_optimizer_.step()
        with torch.no_grad():
            min_v = torch.min(v_pi_1,v_pi_2) 
            return safe_value_loss.item(), safe_small_q_loss.item(), torch.min(constraint_batch - min_v, scalar + self.safety_lambda*(constraint_batch - min_v)).mean().item()
        
    def ScheduleParams(self):
        t = max(self.update_count_ - self.safety_lambda_start_to_decrease_, 0)
        t_ratio = min(1.0, t / self.safety_lambda_decrease_interval_)
        power = 5
        poly_decay = (1.0 - t_ratio)**power
        self.safety_lambda = self.safety_lambda_goal_ + (self.safety_lambda_init_ - self.safety_lambda_goal_) * poly_decay

        lr_decay_start = self.safety_lambda_start_to_decrease_ + self.safety_lambda_decrease_interval_/2.0
        t_lr = max(self.update_count_ - lr_decay_start, 0)
        t_ratio = min(1.0, t_lr / self.lr_decrease_interval_)
        lr_poly_decay = (1.0 - t_ratio)**power

        self.safe_value_optimizer_.param_groups[0]['lr'] = self.lr_value_min_ + (self.lr_value_max_ - self.lr_value_min_) * lr_poly_decay
        self.safe_q_optimizer_.param_groups[0]['lr'] = self.lr_coeff_min_ + (self.lr_coeff_max_ - self.lr_coeff_min_) * lr_poly_decay
        self.safe_q_optimizer_.param_groups[1]['lr'] = self.lr_scalar_min_ + (self.lr_scalar_max_ - self.lr_scalar_min_) * lr_poly_decay

    def update(self):
        self.update_count_ += 1

        if (self.update_count_-self.warmup_steps_ > 0):
            self.ScheduleParams()
            if self.warm_up_:
                self.warm_up_ = False
        else:
            self.safe_value_optimizer_.param_groups[0]['lr'] = self.lr_value_max_ * self.update_count_ / self.warmup_steps_
            self.safe_q_optimizer_.param_groups[0]['lr'] = self.lr_coeff_max_ * self.update_count_ / self.warmup_steps_
            self.safe_q_optimizer_.param_groups[1]['lr'] = self.lr_scalar_max_ * self.update_count_ / self.warmup_steps_
        
        survive_sample_size = self.batch_size 

        batch = self.replay_memory.sample(survive_sample_size)
        state_batch, action_batch, constraint_batch, next_state_batch, next_constraint_batch, _ = map(np.stack, zip(*batch))
        state_batch = torch.tensor(state_batch, device=self.device_, dtype=self.dtype_)
        next_state_batch = torch.tensor(next_state_batch, device=self.device_, dtype=self.dtype_)
        action_batch = torch.tensor(action_batch, device=self.device_, dtype=self.dtype_)
        constraint_batch = torch.tensor(constraint_batch, device=self.device_, dtype=self.dtype_).unsqueeze(-1)
        next_constraint_batch = torch.tensor(next_constraint_batch, device=self.device_, dtype=self.dtype_).unsqueeze(-1)
        
        constraint_batch /= self.c_max_
        next_constraint_batch /= self.c_max_
        safe_value_loss, safe_small_q_loss, consistency_loss = self.safety_critic_update(state_batch, action_batch, constraint_batch, next_state_batch, next_constraint_batch)
                
        self.soft_update()
        
        return safe_value_loss, safe_small_q_loss, consistency_loss

    def append_transition(self, state, action, constraint, next_state, next_constraint, fail):
        self.replay_memory.push_two_signals(state, action, constraint, next_state, next_constraint, fail)
        
    def TargetUpdateHelper(self, cur:SafeValues, target:SafeValues, tau_:float):
        target_net_state_dict = target.state_dict()
        cur_net_state_dict = cur.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = cur_net_state_dict[key]*tau_ + target_net_state_dict[key]*(1-tau_)
        target.load_state_dict(target_net_state_dict)
        
    def soft_update(self):
        self.TargetUpdateHelper(self.SafeValues_, self.TargetSafeValues_, self.tau_safe_)
        
    def hard_update(self):
        self.TargetUpdateHelper(self.SafeValues_, self.TargetSafeValues_, 1)

    def NNsave(self,dir:str):
        safe_values_to_save = self.SafeValues_._orig_mod if hasattr(self.SafeValues_, '_orig_mod') else self.SafeValues_
        target_safe_values_to_save = self.TargetSafeValues_._orig_mod if hasattr(self.TargetSafeValues_, '_orig_mod') else self.TargetSafeValues_
        ckpt = {
            'SafeValues_state':    safe_values_to_save.state_dict(),
            'TargetSafeValues_state':    target_safe_values_to_save.state_dict(),
            'safe_value_optimizer_state':    self.safe_value_optimizer_.state_dict(),
            'safe_q_optimizer_state':    self.safe_q_optimizer_.state_dict(),
            'c_max': self.c_max_,
            }
        torch.save(ckpt, dir)
        
    def NNload(self, dir: str, eval = True):
        ckpt = torch.load(dir, map_location=self.device_, weights_only=False)

        # (2) weight 로드
        self.SafeValues_.load_state_dict(ckpt['SafeValues_state'])
        self.SafeValues_.to(self.device_)
        self.TargetSafeValues_.load_state_dict(ckpt['TargetSafeValues_state'])
        self.c_max_ = ckpt['c_max']
        self.TargetSafeValues_.to(self.device_)
        if not eval:
            # (3) 옵티마이저 로드
            self.safe_value_optimizer_.load_state_dict(ckpt['safe_value_optimizer_state'])
            self.safe_q_optimizer_.load_state_dict(ckpt['safe_q_optimizer_state'])

            # 옵티마이저 내부 텐서도 device 맞춰줄 필요 있습니다:
            for state in self.safe_value_optimizer_.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device_)
            for state in self.safe_q_optimizer_.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device_)

    def save_replay_memory(self, dir_name):
        self.replay_memory.save_buffer(dir_name, add_arg="/buffer")
    def load_replay_memory(self, dir_name):
        self.replay_memory.load_buffer(dir_name, add_arg="/buffer")
    def load_state_dict(self, target, load_cur = False):
        self.TargetSafeValues_.load_state_dict(target.TargetSafeValues_.state_dict())
        self.safety_lambda = deepcopy( target.safety_lambda )
        self.c_max_ = deepcopy(target.c_max_)
        if load_cur:
            self.SafeValues_.load_state_dict(target.SafeValues_.state_dict())

    def GetRMLength(self):
        return len(self.replay_memory)
    