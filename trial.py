import numpy as np
from biomind.params import TRIAL_DEFAULTS


class TrialManager:
    """
    Manages the 3-phase trial loop:
      Phase 0: Stimulus presentation, waiting for thalamic threshold crossing
      Phase 1: Post-decision movement delay
      Phase 2: Reward delivery + inter-trial interval
    """
    
    def __init__(self, agent, pop_index, n_actions=2, reward_schedule=None,
                 qlearner=None, **kwargs):
        """
        Args:
            agent: Agent object
            pop_index: dict mapping (name, channel) -> population index
            n_actions: number of action channels
            reward_schedule: array [n_trials, n_actions] of reward values
            qlearner: QLearner object
            **kwargs: override any TRIAL_DEFAULTS
        """
        self.agent = agent
        self.pop_index = pop_index
        self.n_actions = n_actions
        self.reward_schedule = reward_schedule
        self.qlearner = qlearner
        
        # Load trial parameters with overrides
        self.params = {**TRIAL_DEFAULTS, **kwargs}
        
        # Identify cortex and thalamus population indices
        self.cx_popids = []
        self.th_popids = []
        self.dspn_popids = []
        self.ispn_popids = []
        self.str_popids = []
        
        for ch in range(n_actions):
            self.cx_popids.append(pop_index[('Cx', ch)])
            self.th_popids.append(pop_index[('Th', ch)])
            self.dspn_popids.append(pop_index[('dSPN', ch)])
            self.ispn_popids.append(pop_index[('iSPN', ch)])
        
        self.str_popids = self.dspn_popids + self.ispn_popids
        
        # Trial state
        self.phase = 0
        self.phase_timer = 0
        self.global_timer = 0
        self.trial_num = 0
        self.chosen_action = None
        self.motor_queued = None
        self.gain = np.ones(n_actions)
        
        # Results storage
        self.results = []
    
    def step(self):
        """
        Advance the trial state machine by 1ms (= 5 timesteps at dt=0.2ms).
        
        Returns:
            trial_complete: bool, True if a trial just finished
        """
        a = self.agent
        trial_complete = False
        
        # Apply stimulus
        if self.phase == 0:
            stim = self.gain * self.params['max_stim']
            for ch in range(self.n_actions):
                popid = self.cx_popids[ch]
                a.FreqExt_AMPA[popid] = (
                    a.FreqExt_AMPA_basestim[popid] 
                    + np.ones(a.pops[popid]['N']) * stim[ch]
                )

        #  Run 5 neural timesteps (1 ms)
        from biomind.timestep import multi_timestep
        multi_timestep(a, 5)
        
        self.phase_timer += 1
        self.global_timer += 1
        
        # === Phase 0: Check for decision ===
        if self.phase == 0:
            # Compute thalamic firing rates
            th_rates = np.array([
                a.rollingbuffer[popid].mean() / a.pops[popid]['N'] / a.dt * 1000.0
                for popid in self.th_popids
            ])
            
            threshold = self.params['thalamic_threshold']
            crossed = np.where(th_rates > threshold)[0]
            
            if len(crossed) > 0 or self.phase_timer > self.params['choice_timeout']:
                self.phase = 1
                self.phase_timer = 0
                
                if len(crossed) > 0:
                    self.motor_queued = crossed[0]  # first past the post
                    self.gain = np.zeros(self.n_actions)
                    self.gain[self.motor_queued] = self.params['sustained_fraction']
                else:
                    self.motor_queued = -1
                    self.gain = np.zeros(self.n_actions)
        
        # Phase 1: Movement delay
        elif self.phase == 1:
            if self.phase_timer > self.params['movement_time']:
                self.phase = 2
                self.phase_timer = 0
                self.gain = np.zeros(self.n_actions)
                
                # Determine chosen action
                if self.motor_queued is not None and self.motor_queued >= 0:
                    self.chosen_action = self.motor_queued
                else:
                    self.chosen_action = -1
                
                # Deliver reward and compute dopamine
                if self.qlearner is not None and self.reward_schedule is not None:
                    reward = self.qlearner.get_reward(
                        self.reward_schedule, self.chosen_action, self.trial_num
                    )
                    DA_phasic = self.qlearner.update(self.chosen_action, reward)
                    
                    # Inject dopamine into striatal populations
                    for popid in self.str_popids:
                        a.dpmn_DAp[popid] *= 0
                        a.dpmn_DAp[popid] += DA_phasic
                    
                    self.results.append({
                        'trial': self.trial_num,
                        'chosen_action': self.chosen_action,
                        'reward': reward,
                        'DA_phasic': DA_phasic,
                        'Q_values': self.qlearner.Q.copy(),
                    })
        
        # Phase 2: Inter-trial interval
        elif self.phase == 2:
            if self.phase_timer > self.params['inter_trial_interval']:
                self.trial_num += 1
                self.phase = 0
                self.phase_timer = 0
                self.gain = np.ones(self.n_actions)
                self.chosen_action = None
                self.motor_queued = None
                trial_complete = True
        
        return trial_complete
