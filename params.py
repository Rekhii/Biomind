import numpy as np


# SIMULATION
DT = 0.2          # ms, integration timestep
BUFFER_LEN = 300  # rolling window for firing rate (300 * 0.2ms = 60ms)


# DEFAULT NEURON (Leaky Integrate-and-Fire + calcium + K+ channels)
NEURON_DEFAULTS = {
    'N':          75,      # neurons per population per channel
    'C':          0.5,     # nF, membrane capacitance
    'Taum':       20.0,    # ms, membrane time constant
    'RestPot':    -70.0,   # mV, resting membrane potential
    'ResetPot':   -55.0,   # mV, post-spike reset potential
    'Threshold':  -50.0,   # mV, spike threshold
    # Calcium dynamics (afterhyperpolarization)
    'RestPot_ca': -85.0,   # mV, calcium reversal potential
    'Alpha_ca':   0.5,     # uM, calcium increment per spike
    'Tau_ca':     80.0,    # ms, calcium decay time constant
    'Eff_ca':     0.0,     # calcium efficacy (unused in default)
    # Low-threshold calcium (T-current, for rebound bursting)
    'tauhm':      20.0,    # ms, burst inactivation time constant
    'tauhp':      100.0,   # ms, de-inactivation time constant
    'V_h':        -60.0,   # mV, burst activation threshold
    'V_T':        120.0,   # mV, T-current reversal potential
    'g_T':        0.0,     # mS/cm2, T-current max conductance (OFF by default)
    # Anomalous delayed rectifier
    'g_adr_max':  0.0,     # max conductance (OFF by default)
    'Vadr_h':     -100.0,  # mV, half-activation voltage
    'Vadr_s':     10.0,    # mV, slope
    'ADRRevPot':  -90.0,   # mV, reversal potential
    # Outward rectifying K+
    'g_k_max':    0.0,     # max conductance (OFF by default)
    'Vk_h':       -34.0,   # mV, half-activation
    'Vk_s':       6.5,     # mV, slope
    'tau_k_max':  8.0,     # ms, max time constant
    'n_k':        0.0,     # gating variable initial value
    'h':          1.0,     # T-current gating variable initial value
}

# POPULATION-SPECIFIC OVERRIDES
# Each nucleus overrides specific defaults based on its biophysics.
POP_SPECIFIC = {
    'Cx':   {'N': 204, 'dpmn_cortex': 1},         # Cortex: largest, flags dopamine source
    'CxI':  {'N': 186, 'C': 0.2, 'Taum': 10.0},   # Cortical interneurons: small, fast
    'dSPN': {},                                      # D1-MSNs: use all defaults
    'iSPN': {},                                      # D2-MSNs: use all defaults
    'FSI':  {'C': 0.2, 'Taum': 10.0},              # Fast-spiking interneurons: small, fast
    'GPe':  {'N': 750, 'g_T': 0.06, 'Taum': 20.0}, # GPe: pacemaker, rebound bursting ON
    'STN':  {'N': 750, 'g_T': 0.06},               # STN: rebound bursting ON
    'GPi':  {},                                      # GPi: use defaults (output nucleus)
    'Th':   {'Taum': 27.78},                        # Thalamus: slower integration
}

# Canonical population order (matches CBGTPy indexing)
POP_NAMES = ['GPi', 'STN', 'GPe', 'dSPN', 'iSPN', 'Cx', 'Th', 'FSI', 'CxI']


# RECEPTOR KINETICS
RECEPTOR_DEFAULTS = {
    'Tau_AMPA':     2.0,    # ms, AMPA decay time constant
    'RevPot_AMPA':  0.0,    # mV, AMPA reversal potential
    'Tau_GABA':     5.0,    # ms, GABA-A decay time constant
    'RevPot_GABA':  -70.0,  # mV, GABA reversal potential
    'Tau_NMDA':     100.0,  # ms, NMDA decay time constant
    'RevPot_NMDA':  0.0,    # mV, NMDA reversal potential
}

# NMDA saturation constant (from Jahr & Stevens 1990)
NMDA_ALPHA = 0.6332

# NMDA Mg2+ block parameters: 1 / (1 + exp(-0.062 * V / 3.57))
NMDA_MG_BLOCK_SLOPE = -0.062
NMDA_MG_BLOCK_SCALE = 3.57


# EXTERNAL BACKGROUND INPUT
# Models the thousands of cortical synapses bombarding each nucleus.
# FreqExt = input firing rate (kHz), MeanExtEff = conductance, MeanExtCon = N connections
BASESTIM = {
    'Cx':   {'FreqExt_AMPA': 2.3,  'MeanExtEff_AMPA': 2.0,  'MeanExtCon_AMPA': 800},
    'CxI':  {'FreqExt_AMPA': 3.7,  'MeanExtEff_AMPA': 1.2,  'MeanExtCon_AMPA': 640},
    'dSPN': {'FreqExt_AMPA': 1.3,  'MeanExtEff_AMPA': 4.0,  'MeanExtCon_AMPA': 800},
    'iSPN': {'FreqExt_AMPA': 1.3,  'MeanExtEff_AMPA': 4.0,  'MeanExtCon_AMPA': 800},
    'FSI':  {'FreqExt_AMPA': 3.6,  'MeanExtEff_AMPA': 1.55, 'MeanExtCon_AMPA': 800},
    'GPe':  {'FreqExt_AMPA': 4.0,  'MeanExtEff_AMPA': 2.0,  'MeanExtCon_AMPA': 800,
             'FreqExt_GABA': 2.0,  'MeanExtEff_GABA': 2.0,  'MeanExtCon_GABA': 2000},
    'STN':  {'FreqExt_AMPA': 4.45, 'MeanExtEff_AMPA': 1.65, 'MeanExtCon_AMPA': 800},
    'GPi':  {'FreqExt_AMPA': 0.8,  'MeanExtEff_AMPA': 5.9,  'MeanExtCon_AMPA': 800},
    'Th':   {'FreqExt_AMPA': 2.2,  'MeanExtEff_AMPA': 2.5,  'MeanExtCon_AMPA': 800},
}


# PATHWAY CONNECTIVITY
# Each row: (src, dest, receptor, type, connection_prob, efficacy, plastic)
PATHWAYS = [
    # DIRECT PATHWAY (Go)
    ('Cx',   'dSPN', 'AMPA', 'syn', 1.0,    0.015,   True),   # PLASTIC: cortex teaches D1
    ('Cx',   'dSPN', 'NMDA', 'syn', 1.0,    0.02,    False),
    ('dSPN', 'GPi',  'GABA', 'syn', 1.0,    2.09,    False),  # disinhibition gate

    # INDIRECT PATHWAY (NoGo)
    ('Cx',   'iSPN', 'AMPA', 'syn', 1.0,    0.015,   True),   # PLASTIC: cortex teaches D2
    ('Cx',   'iSPN', 'NMDA', 'syn', 1.0,    0.02,    False),
    ('iSPN', 'GPe',  'GABA', 'syn', 1.0,    4.07,    False),
    ('GPe',  'STN',  'GABA', 'syn', 0.0667, 0.35,    False),
    ('STN',  'GPi',  'NMDA', 'all', 1.0,    0.038,   False),  # broad brake

    # GPe SELF-INHIBITION + FEEDBACK
    ('GPe',  'GPe',  'GABA', 'all', 0.0667, 1.75,    False),
    ('GPe',  'GPi',  'GABA', 'syn', 1.0,    0.058,   False),
    ('STN',  'GPe',  'AMPA', 'syn', 0.1617, 0.07,    False),
    ('STN',  'GPe',  'NMDA', 'syn', 0.1617, 1.51,    False),

    # OUTPUT: GPi -> Thalamus
    ('GPi',  'Th',   'GABA', 'syn', 1.0,    0.3315,  False),

    # THALAMO-CORTICAL FEEDBACK
    ('Th',   'Cx',   'NMDA', 'all', 0.8334, 0.03,    False),
    ('Th',   'dSPN', 'AMPA', 'syn', 1.0,    0.3825,  False),
    ('Th',   'iSPN', 'AMPA', 'syn', 1.0,    0.3825,  False),
    ('Th',   'FSI',  'AMPA', 'all', 0.8334, 0.1,     False),
    ('Th',   'CxI',  'NMDA', 'all', 0.8334, 0.015,   False),

    # STRIATAL LATERAL INHIBITION
    ('dSPN', 'dSPN', 'GABA', 'syn', 0.45,   0.28,    False),
    ('dSPN', 'iSPN', 'GABA', 'syn', 0.45,   0.28,    False),
    ('iSPN', 'iSPN', 'GABA', 'syn', 0.45,   0.28,    False),
    ('iSPN', 'dSPN', 'GABA', 'syn', 0.50,   0.28,    False),

    # FSI FEEDFORWARD INHIBITION
    ('FSI',  'FSI',  'GABA', 'all', 1.0,    3.2583,  False),
    ('FSI',  'dSPN', 'GABA', 'all', 1.0,    1.2,     False),
    ('FSI',  'iSPN', 'GABA', 'all', 1.0,    1.1,     False),
    ('Cx',   'FSI',  'AMPA', 'all', 1.0,    0.19,    False),

    # CORTICAL RECURRENCE
    ('Cx',   'Cx',   'AMPA', 'syn', 0.13,   0.0127,  False),
    ('Cx',   'Cx',   'NMDA', 'syn', 0.13,   0.08,    False),
    ('Cx',   'CxI',  'AMPA', 'all', 0.0725, 0.113,   False),
    ('Cx',   'CxI',  'NMDA', 'all', 0.0725, 0.525,   False),
    ('CxI',  'Cx',   'GABA', 'all', 0.5,    1.05,    False),
    ('CxI',  'CxI',  'GABA', 'all', 1.0,    1.075,   False),

    # CORTEX-THALAMUS
    ('Cx',   'Th',   'AMPA', 'syn', 1.0,    0.025,   False),
    ('Cx',   'Th',   'NMDA', 'syn', 1.0,    0.029,   False),
]


# DOPAMINE / PLASTICITY PARAMETERS
DPMN_DEFAULTS = {
    'dpmn_tauDOP':   2.0,     # ms, dopamine decay
    'dpmn_dPRE':     0.8,     # pre-synaptic trace increment
    'dpmn_dPOST':    0.04,    # post-synaptic trace increment
    'dpmn_tauE':     100.0,   # ms, eligibility trace decay
    'dpmn_tauPRE':   15.0,    # ms, pre-synaptic trace decay
    'dpmn_tauPOST':  6.0,     # ms, post-synaptic trace decay
    'dpmn_DAt':      0.0,     # tonic dopamine level
    'dpmn_taum':     1e100,   # motivation decay (effectively infinite = no decay)
    'dpmn_m':        1.0,     # motivation level
    # fDA nonlinearity parameters
    'dpmn_x_fda':    0.5,     # DA threshold for saturation
    'dpmn_y_fda':    3.0,     # fDA saturation value
    'dpmn_d2_DA_eps': 0.3,    # D2 sensitivity scaling (D2 < D1)
}

# D1-SPN (direct pathway) specific
D1_PARAMS = {
    'dpmn_type':    1,        # 1 = D1
    'dpmn_alphaw':  39.5,     # learning rate (positive = LTP on reward)
    'dpmn_wmax':    0.055,    # max synaptic weight
    'dpmn_a':       1.0,      # fDA parameter
    'dpmn_b':       0.1,      # fDA parameter
    'dpmn_c':       0.05,     # fDA parameter
}

# D2-SPN (indirect pathway) specific
D2_PARAMS = {
    'dpmn_type':    2,        # 2 = D2
    'dpmn_alphaw':  -38.2,    # learning rate (negative = LTD on reward)
    'dpmn_wmax':    0.035,    # max synaptic weight
    'dpmn_a':       0.5,      # fDA parameter
    'dpmn_b':       0.005,    # fDA parameter
    'dpmn_c':       0.05,     # fDA parameter
}


# Q-LEARNING / REWARD PARAMETERS
Q_PARAMS = {
    'q_alpha':  0.1,    # Q-value learning rate
    'C_scale':  80.0,   # RPE-to-dopamine scaling factor
    'Q_init':   0.5,    # initial Q-value for all actions
}


# TRIAL PARAMETERS
TRIAL_DEFAULTS = {
    'n_actions':           2,       # number of action channels
    'thalamic_threshold':  30.0,    # Hz, firing rate threshold for decision
    'choice_timeout':      1000,    # ms, max decision time
    'movement_time':       300,     # ms, post-decision delay before reward
    'inter_trial_interval': 300,    # ms, gap between trials
    'sustained_fraction':  0.3,     # gain on chosen action post-decision
    'max_stim':            3.0,     # max cortical stimulus strength
    'warmup_steps':        5000,    # timesteps to stabilize before trials
}


# REFRACTORY PERIOD
REFRACTORY_STEPS = 10  # number of dt steps in refractory period (10 * 0.2ms = 2ms)
