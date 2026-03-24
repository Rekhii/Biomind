import numpy as np
from biomind.params import Q_PARAMS


class QLearner:
    """
    Tracks Q-values and converts reward prediction errors to dopamine signals.
    """
    
    def __init__(self, n_actions=2):
        self.n_actions = n_actions
        self.alpha = Q_PARAMS['q_alpha']       # Q-learning rate
        self.C_scale = Q_PARAMS['C_scale']     # RPE-to-dopamine gain
        self.Q = np.full(n_actions, Q_PARAMS['Q_init'])  # Q-values
        self.Q_history = [self.Q.copy()]
        self.DA_history = []
    
    def get_reward(self, reward_schedule, chosen_action, trial_num):
        """
        Get reward value from the reward schedule.
        
        Args:
            reward_schedule: array-like, reward_schedule[trial][action] = reward value
            chosen_action: int, index of chosen action (or -1 for no choice)
            trial_num: int, current trial number
            
        Returns:
            reward: float, the reward value
        """
        if chosen_action < 0:
            return 0.0
        return reward_schedule[trial_num][chosen_action]
    
    def update(self, chosen_action, reward):
        """
        Update Q-values and compute dopamine signal.
        
        Args:
            chosen_action: int, index of chosen action (or -1 for no choice)
            reward: float, reward received
            
        Returns:
            DA_phasic: float, phasic dopamine signal (scaled RPE)
        """
        if chosen_action < 0 or chosen_action >= self.n_actions:
            self.Q_history.append(self.Q.copy())
            self.DA_history.append(0.0)
            return 0.0
        
        # Prediction error
        RPE = reward - self.Q[chosen_action]
        
        # Q-value update
        self.Q[chosen_action] += self.alpha * RPE
        
        # Dopamine signal = scaled RPE
        DA_phasic = RPE * self.C_scale
        
        self.Q_history.append(self.Q.copy())
        self.DA_history.append(DA_phasic)
        
        return DA_phasic
