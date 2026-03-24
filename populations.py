import numpy as np
from biomind.params import (
    NEURON_DEFAULTS, POP_SPECIFIC, POP_NAMES, RECEPTOR_DEFAULTS,
    BASESTIM, PATHWAYS, DPMN_DEFAULTS, D1_PARAMS, D2_PARAMS
)


def build_population_data(n_actions=2):
    """
    Build a list of population dicts, one per population per action channel.
    
    For 2 actions and 9 base populations:
      - Channel-specific pops (GPi, STN, GPe, dSPN, iSPN, Cx, Th) get duplicated
        -> 7 * 2 = 14 populations
      - Non-channel pops (FSI, CxI) stay single
        -> 2 populations
      - Total: 16 populations for 2 actions
    
    Returns:
        pops: list of dicts, each with all neuron/receptor/stim/dpmn params
        pop_index: dict mapping (name, channel) -> index in pops list
    """
    # Which populations are channel-specific vs shared
    channel_pops = ['GPi', 'STN', 'GPe', 'dSPN', 'iSPN', 'Cx', 'Th']
    shared_pops = ['FSI', 'CxI']
    
    pops = []
    pop_index = {}
    
    for name in POP_NAMES:
        if name in channel_pops:
            for ch in range(n_actions):
                pop = _build_single_pop(name, ch)
                idx = len(pops)
                pops.append(pop)
                pop_index[(name, ch)] = idx
        else:
            pop = _build_single_pop(name, -1)  # -1 = shared across channels
            idx = len(pops)
            pops.append(pop)
            pop_index[(name, -1)] = idx
    
    return pops, pop_index


def _build_single_pop(name, channel):
    """Build parameter dict for a single population."""
    pop = {}
    pop['name'] = name
    pop['channel'] = channel
    
    # Start with neuron defaults
    for k, v in NEURON_DEFAULTS.items():
        pop[k] = v
    
    # Apply population-specific overrides
    if name in POP_SPECIFIC:
        for k, v in POP_SPECIFIC[name].items():
            pop[k] = v
    
    # Receptor kinetics
    for k, v in RECEPTOR_DEFAULTS.items():
        pop[k] = v
    
    # Background stimulation
    if name in BASESTIM:
        for k, v in BASESTIM[name].items():
            pop[k] = v
    
    # Fill missing stim params with 0
    for receptor in ['AMPA', 'GABA', 'NMDA']:
        for prefix in ['FreqExt_', 'MeanExtEff_', 'MeanExtCon_']:
            key = prefix + receptor
            if key not in pop:
                pop[key] = 0.0
    
    # Dopamine params for SPNs
    pop['dpmn_type'] = 0  # default: not dopamine-modulated
    pop['dpmn_cortex'] = pop.get('dpmn_cortex', 0)
    
    if name == 'dSPN':
        for k, v in DPMN_DEFAULTS.items():
            pop[k] = v
        for k, v in D1_PARAMS.items():
            pop[k] = v
    elif name == 'iSPN':
        for k, v in DPMN_DEFAULTS.items():
            pop[k] = v
        for k, v in D2_PARAMS.items():
            pop[k] = v
    
    return pop


def build_connectivity(pops, pop_index, n_actions=2):
    """
    Build connectivity, efficacy, and plasticity matrices for each receptor.
    
    Returns:
        conn: dict with keys 'AMPA', 'GABA', 'NMDA'
              each value is (connection_matrix, efficacy_matrix, plasticity_matrix)
              where matrices are lists-of-lists [src][dest] of either None or 2D numpy arrays
    """
    n_pops = len(pops)
    
    conn = {}
    for receptor in ['AMPA', 'GABA', 'NMDA']:
        con_matrix = [[None for _ in range(n_pops)] for _ in range(n_pops)]
        eff_matrix = [[None for _ in range(n_pops)] for _ in range(n_pops)]
        plastic_matrix = [[False for _ in range(n_pops)] for _ in range(n_pops)]
        
        for pathway in PATHWAYS:
            src_name, dest_name, rec, ptype, prob, eff, plastic = pathway
            
            if rec != receptor:
                continue
            
            # Get all (src_idx, dest_idx) pairs for this pathway
            pairs = _get_population_pairs(
                src_name, dest_name, ptype, pop_index, n_actions
            )
            
            for src_idx, dest_idx in pairs:
                n_src = pops[src_idx]['N']
                n_dest = pops[dest_idx]['N']
                
                # Connection matrix: probabilistic or full
                if prob >= 1.0:
                    con_data = np.ones((n_src, n_dest))
                else:
                    con_data = (np.random.rand(n_src, n_dest) < prob).astype(float)
                
                # Efficacy matrix: connection * weight
                eff_data = con_data * eff
                
                con_matrix[src_idx][dest_idx] = con_data
                eff_matrix[src_idx][dest_idx] = eff_data
                plastic_matrix[src_idx][dest_idx] = plastic
        
        conn[receptor] = (con_matrix, eff_matrix, plastic_matrix)
    
    return conn


def _get_population_pairs(src_name, dest_name, ptype, pop_index, n_actions):
    """
    Get (src_idx, dest_idx) pairs based on pathway type.
    
    'syn' = channel-specific: action 1 src -> action 1 dest only
    'all' = all-to-all: every src channel -> every dest channel
    """
    pairs = []
    
    src_indices = _get_pop_indices(src_name, pop_index, n_actions)
    dest_indices = _get_pop_indices(dest_name, pop_index, n_actions)
    
    if ptype == 'syn':
        # Channel-matched only
        for src_idx, src_ch in src_indices:
            for dest_idx, dest_ch in dest_indices:
                if src_ch == dest_ch or src_ch == -1 or dest_ch == -1:
                    pairs.append((src_idx, dest_idx))
    elif ptype == 'all':
        # All-to-all
        for src_idx, _ in src_indices:
            for dest_idx, _ in dest_indices:
                pairs.append((src_idx, dest_idx))
    
    return pairs


def _get_pop_indices(name, pop_index, n_actions):
    """Get list of (index, channel) for a population name."""
    indices = []
    # Try channel-specific first
    for ch in range(n_actions):
        key = (name, ch)
        if key in pop_index:
            indices.append((pop_index[key], ch))
    # Try shared
    key = (name, -1)
    if key in pop_index:
        indices.append((pop_index[key], -1))
    return indices
