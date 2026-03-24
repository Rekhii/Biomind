"""
BioMind-BG: agent.py
The complete agent state. All neural variables as NumPy arrays.
No classes wrapping classes. Just a simple namespace with arrays.
"""

import numpy as np
from biomind.params import DT, BUFFER_LEN, REFRACTORY_STEPS


class Agent:
    """
    Holds all state variables for the CBGT network simulation.
    Every variable is a list of numpy arrays, indexed by population id.
    """
    
    def __init__(self, pops, connectivity):
        """
        Initialize all state arrays from population data and connectivity.
        
        Args:
            pops: list of population dicts (from build_population_data)
            connectivity: dict of (con, eff, plastic) per receptor (from build_connectivity)
        """
        self.dt = DT
        self.n_pops = len(pops)
        self.pops = pops
        
        # Neuron state variables
        self.V = [np.ones(p['N']) * p['RestPot'] for p in pops]
        self.Ca = [np.zeros(p['N']) for p in pops]
        self.h = [np.ones(p['N']) * p['h'] for p in pops]
        self.n_k = [np.ones(p['N']) * p['n_k'] for p in pops]
        self.RefrState = [np.zeros(p['N'], dtype=int) for p in pops]
        self.spikes = [[] for _ in pops]
        self.timesincelastspike = [np.zeros(p['N']) for p in pops]
        self.Ptimesincelastspike = [np.zeros(p['N']) for p in pops]
        
        # Neuron parameters (expanded to per-neuron arrays)
        self.N = [np.full(p['N'], p['N']) for p in pops]
        self.C = [np.full(p['N'], p['C']) for p in pops]
        self.Taum = [np.full(p['N'], p['Taum']) for p in pops]
        self.RestPot = [np.full(p['N'], p['RestPot']) for p in pops]
        self.ResetPot = [np.full(p['N'], p['ResetPot']) for p in pops]
        self.Threshold = [np.full(p['N'], p['Threshold']) for p in pops]
        self.alpha_ca = [np.full(p['N'], p['Alpha_ca']) for p in pops]
        self.Tau_ca = [np.full(p['N'], p['Tau_ca']) for p in pops]
        self.V_h = [np.full(p['N'], p['V_h']) for p in pops]
        self.V_T = [np.full(p['N'], p['V_T']) for p in pops]
        self.g_T = [np.full(p['N'], p['g_T']) for p in pops]
        self.tauhm = [np.full(p['N'], p['tauhm']) for p in pops]
        self.tauhp = [np.full(p['N'], p['tauhp']) for p in pops]
        self.g_adr_max = [np.full(p['N'], p['g_adr_max']) for p in pops]
        self.Vadr_h = [np.full(p['N'], p['Vadr_h']) for p in pops]
        self.Vadr_s = [np.full(p['N'], p['Vadr_s']) for p in pops]
        self.ADRRevPot = [np.full(p['N'], p['ADRRevPot']) for p in pops]
        self.g_k_max = [np.full(p['N'], p['g_k_max']) for p in pops]
        self.Vk_h = [np.full(p['N'], p['Vk_h']) for p in pops]
        self.Vk_s = [np.full(p['N'], p['Vk_s']) for p in pops]
        self.tau_k_max = [np.full(p['N'], p['tau_k_max']) for p in pops]
        
        # AHP and K reversal (CBGTPy uses defaults of 0)
        self.g_ahp = [np.zeros(p['N']) for p in pops]
        self.Vk = [np.full(p['N'], -90.0) for p in pops]  # K+ reversal
        
        # Receptor parameters (per-neuron)
        for receptor in ['AMPA', 'GABA', 'NMDA']:
            tau_key = f'Tau_{receptor}'
            rev_key = f'RevPot_{receptor}'
            setattr(self, tau_key, [np.full(p['N'], p[tau_key]) for p in pops])
            setattr(self, rev_key, [np.full(p['N'], p[rev_key]) for p in pops])

        # Synaptic conductance state

        # External (background) conductances
        self.ExtS_AMPA = [np.zeros(p['N']) for p in pops]
        self.ExtS_GABA = [np.zeros(p['N']) for p in pops]
        self.ExtS_NMDA = [np.zeros(p['N']) for p in pops]
        self.ExtMuS_AMPA = [np.zeros(p['N']) for p in pops]
        self.ExtMuS_GABA = [np.zeros(p['N']) for p in pops]
        self.ExtMuS_NMDA = [np.zeros(p['N']) for p in pops]
        self.ExtSigmaS_AMPA = [np.zeros(p['N']) for p in pops]
        self.ExtSigmaS_GABA = [np.zeros(p['N']) for p in pops]
        self.ExtSigmaS_NMDA = [np.zeros(p['N']) for p in pops]
        
        # Lateral (network) conductances
        self.LS_AMPA = [np.zeros(p['N']) for p in pops]
        self.LS_GABA = [np.zeros(p['N']) for p in pops]
        self.LS_NMDA = [np.zeros(p['N']) for p in pops]
        
        # External input parameters (per-neuron)
        for receptor in ['AMPA', 'GABA', 'NMDA']:
            for prefix in ['FreqExt_', 'MeanExtEff_', 'MeanExtCon_']:
                key = prefix + receptor
                setattr(self, key, [np.full(p['N'], p.get(key, 0.0)) for p in pops])
        
        # Store basestim for cortex (needed for stimulus modulation)
        self.FreqExt_AMPA_basestim = [arr.copy() for arr in self.FreqExt_AMPA]
        
        # Optogenetic (unused for now, kept for compatibility)
        self.ExtS_Opto = [np.zeros(p['N']) for p in pops]
        self.RevPot_ChR2 = [np.zeros(p['N']) for p in pops]
        self.RevPot_NpHR = [np.full(p['N'], -400.0) for p in pops]
        
        #  Connectivity matrices (from build_connectivity)
        self.AMPA_con, self.AMPA_eff, self.AMPA_plastic = connectivity['AMPA']
        self.GABA_con, self.GABA_eff, _ = connectivity['GABA']
        self.NMDA_con, self.NMDA_eff, _ = connectivity['NMDA']
        
        # NMDA last conductance tracker (for saturation)
        self.LastConductanceNMDA = self._create_aux_synapse_data(pops, self.NMDA_con)
        
        # Dopamine / plasticity state
        self.dpmn_type = [np.full(p['N'], p.get('dpmn_type', 0)) for p in pops]
        self.dpmn_cortex = [np.full(p['N'], p.get('dpmn_cortex', 0)) for p in pops]
        self.dpmn_DAp = [np.zeros(p['N']) for p in pops]
        self.dpmn_DAt = [np.full(p['N'], p.get('dpmn_DAt', 0.0)) for p in pops]
        self.dpmn_tauDOP = [np.full(p['N'], p.get('dpmn_tauDOP', 2.0)) for p in pops]
        self.dpmn_APRE = [np.zeros(p['N']) for p in pops]
        self.dpmn_APOST = [np.zeros(p['N']) for p in pops]
        self.dpmn_dPRE = [np.full(p['N'], p.get('dpmn_dPRE', 0.8)) for p in pops]
        self.dpmn_dPOST = [np.full(p['N'], p.get('dpmn_dPOST', 0.04)) for p in pops]
        self.dpmn_tauPRE = [np.full(p['N'], p.get('dpmn_tauPRE', 15.0)) for p in pops]
        self.dpmn_tauPOST = [np.full(p['N'], p.get('dpmn_tauPOST', 6.0)) for p in pops]
        self.dpmn_E = [np.zeros(p['N']) for p in pops]
        self.dpmn_tauE = [np.full(p['N'], p.get('dpmn_tauE', 100.0)) for p in pops]
        self.dpmn_m = [np.full(p['N'], p.get('dpmn_m', 1.0)) for p in pops]
        self.dpmn_alphaw = [np.full(p['N'], p.get('dpmn_alphaw', 0.0)) for p in pops]
        self.dpmn_wmax = [np.full(p['N'], p.get('dpmn_wmax', 0.0)) for p in pops]
        self.dpmn_x_fda = [np.full(p['N'], p.get('dpmn_x_fda', 0.5)) for p in pops]
        self.dpmn_y_fda = [np.full(p['N'], p.get('dpmn_y_fda', 3.0)) for p in pops]
        self.dpmn_d2_DA_eps = [np.full(p['N'], p.get('dpmn_d2_DA_eps', 0.3)) for p in pops]
        self.dpmn_XPRE = [np.zeros(p['N']) for p in pops]
        self.dpmn_XPOST = [np.zeros(p['N']) for p in pops]
        self.dpmn_fDA_D1 = [np.zeros(p['N']) for p in pops]
        self.dpmn_fDA_D2 = [np.zeros(p['N']) for p in pops]
        self.dpmn_taum = [np.full(p['N'], p.get('dpmn_taum', 1e100)) for p in pops]
        
        # Temporary computation arrays (avoid re-allocation)
        self.cond = [np.zeros(p['N'], dtype=int) for p in pops]
        self.g_rb = [np.zeros(p['N']) for p in pops]
        self.g_adr = [np.zeros(p['N']) for p in pops]
        self.g_k = [np.zeros(p['N']) for p in pops]
        self.dv = [np.zeros(p['N']) for p in pops]
        self.tau_n = [np.zeros(p['N']) for p in pops]
        self.n_inif = [np.zeros(p['N']) for p in pops]
        self.Vaux = [np.zeros(p['N']) for p in pops]
        
        # Firing rate monitoring
        self.bufferlength = BUFFER_LEN
        self.bufferpointer = 0
        self.rollingbuffer = np.zeros((self.n_pops, BUFFER_LEN))
    
    def _create_aux_synapse_data(self, pops, con_matrix):
        """Create auxiliary NMDA conductance tracking arrays."""
        n = len(pops)
        aux = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if con_matrix[i][j] is not None:
                    aux[i][j] = np.zeros((pops[i]['N'], pops[j]['N']))
        return aux
    
    def get_firing_rates(self):
        """Get current firing rates in Hz for all populations."""
        rates = self.rollingbuffer.mean(axis=1)
        n_neurons = np.array([p['N'] for p in self.pops])
        return rates / n_neurons / self.dt * 1000.0
