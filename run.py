import numpy as np
import time
from biomind.params import TRIAL_DEFAULTS
from biomind.populations import build_population_data, build_connectivity
from biomind.agent import Agent
from biomind.timestep import multi_timestep
from biomind.qlearning import QLearner
from biomind.trial import TrialManager


def run_simulation(n_trials=20, n_actions=2, seed=42, 
                   reward_probs=None, plasticity=True, verbose=True):
    """
    Run a complete BioMind-BG simulation.
    
    Args:
        n_trials: number of trials to run
        n_actions: number of action channels
        seed: random seed
        reward_probs: [n_actions] reward probabilities (default [0.8, 0.2])
        plasticity: whether to enable corticostriatal plasticity
        verbose: print progress
        
    Returns:
        results: dict with trial data, firing rates, weights, etc.
    """
    np.random.seed(seed)
    
    if reward_probs is None:
        reward_probs = [0.8, 0.2]
    
    if verbose:
        print("=" * 60)
        print("BioMind-BG: Basal Ganglia Simulation")
        print("=" * 60)
    

    # 1. BUILD NETWORK
    t0 = time.time()
    
    if verbose:
        print("\n[1/4] Building population data...")
    pops, pop_index = build_population_data(n_actions)
    
    if verbose:
        print(f"  Created {len(pops)} populations:")
        for i, p in enumerate(pops):
            ch_str = f"ch{p['channel']}" if p['channel'] >= 0 else "shared"
            print(f"    [{i:2d}] {p['name']:5s} ({ch_str}): N={p['N']}")
    
    if verbose:
        print("\n[2/4] Building connectivity matrices...")
    connectivity = build_connectivity(pops, pop_index, n_actions)
    
    # Count connections
    n_connections = 0
    for receptor in ['AMPA', 'GABA', 'NMDA']:
        con = connectivity[receptor][0]
        for i in range(len(pops)):
            for j in range(len(pops)):
                if con[i][j] is not None:
                    n_connections += 1
    if verbose:
        print(f"  Total active connections: {n_connections}")
    

    # 2. INITIALIZE AGENT
    if verbose:
        print("\n[3/4] Initializing agent...")
    agent = Agent(pops, connectivity)
    
    if verbose:
        print(f"  dt = {agent.dt} ms")
        print(f"  Buffer length = {agent.bufferlength} ({agent.bufferlength * agent.dt} ms window)")
    

    # 3. WARMUP (stabilize firing rates)
    warmup_steps = TRIAL_DEFAULTS['warmup_steps']
    if verbose:
        print(f"\n  Running warmup ({warmup_steps} steps = {warmup_steps * agent.dt} ms)...")
    
    t1 = time.time()
    
    # During warmup, cortex gets no stimulus (zeroed)
    for ch in range(n_actions):
        popid = pop_index[('Cx', ch)]
        agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
    
    multi_timestep(agent, warmup_steps)
    
    t2 = time.time()
    if verbose:
        print(f"  Warmup complete in {t2-t1:.1f}s")
        print(f"\n  Baseline firing rates (Hz):")
        _print_firing_rates(agent, pops)
    

    # 4. RUN TRIALS
    if verbose:
        print(f"\n[4/4] Running {n_trials} trials...")
        print(f"  Reward probabilities: {reward_probs}")
    
    # Generate reward schedule
    reward_schedule = _generate_reward_schedule(n_trials, reward_probs)
    
    # Q-learner
    qlearner = QLearner(n_actions)
    
    # Trial manager
    trial_mgr = TrialManager(
        agent, pop_index, n_actions,
        reward_schedule=reward_schedule,
        qlearner=qlearner
    )
    
    # Disable plasticity if requested
    if not plasticity:
        for i in range(len(pops)):
            agent.dpmn_type[i] *= 0
    
    # Run trials
    trial_times = []
    t_trial_start = time.time()
    
    while trial_mgr.trial_num < n_trials:
        trial_complete = trial_mgr.step()
        if trial_complete and verbose:
            r = trial_mgr.results[-1]
            elapsed = time.time() - t_trial_start
            print(f"  Trial {r['trial']:3d}: action={r['chosen_action']}, "
                  f"reward={r['reward']:+.1f}, DA={r['DA_phasic']:+.1f}, "
                  f"Q={r['Q_values']}, time={elapsed:.1f}s")
            trial_times.append(elapsed)
            t_trial_start = time.time()
    
    t3 = time.time()
    

    # 5. COLLECT RESULTS
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Simulation complete in {t3-t0:.1f}s total")
        print(f"\nFinal firing rates (Hz):")
        _print_firing_rates(agent, pops)
        
        if len(trial_mgr.results) > 0:
            choices = [r['chosen_action'] for r in trial_mgr.results]
            rewards = [r['reward'] for r in trial_mgr.results]
            print(f"\nBehavioral summary:")
            print(f"  Choices: {choices}")
            print(f"  Total reward: {sum(rewards):.1f}")
            for a_idx in range(n_actions):
                n_chosen = sum(1 for c in choices if c == a_idx)
                print(f"  Action {a_idx}: chosen {n_chosen}/{len(choices)} times "
                      f"({100*n_chosen/max(len(choices),1):.0f}%)")
    
    return {
        'agent': agent,
        'pops': pops,
        'pop_index': pop_index,
        'trial_results': trial_mgr.results,
        'qlearner': qlearner,
        'firing_rates': agent.get_firing_rates(),
    }


def run_baseline_test(seed=42, verbose=True):
    """
    Run a quick baseline test: no trials, just warmup.
    Check that firing rates are biologically plausible.
    
    Expected ranges (from CBGTPy and biology):
        Cx:   2-8 Hz
        dSPN: 0.5-3 Hz (quiet, needs strong input to fire)
        iSPN: 0.5-3 Hz
        FSI:  5-20 Hz
        GPe:  20-60 Hz (autonomous pacemaker)
        STN:  10-30 Hz
        GPi:  40-80 Hz (tonic inhibition of thalamus)
        Th:   2-10 Hz (suppressed by GPi)
    """
    np.random.seed(seed)
    
    if verbose:
        print("=" * 60)
        print("BioMind-BG: Baseline Firing Rate Test")
        print("=" * 60)
    
    pops, pop_index = build_population_data(2)
    connectivity = build_connectivity(pops, pop_index, 2)
    agent = Agent(pops, connectivity)
    
    # Zero cortical stimulus during baseline
    for ch in range(2):
        popid = pop_index[('Cx', ch)]
        agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
    
    if verbose:
        print(f"\nRunning {TRIAL_DEFAULTS['warmup_steps']} warmup steps...")
    
    t0 = time.time()
    multi_timestep(agent, TRIAL_DEFAULTS['warmup_steps'])
    t1 = time.time()
    
    if verbose:
        print(f"Done in {t1-t0:.1f}s\n")
        print("Firing rates (Hz):")
        _print_firing_rates(agent, pops)
    
    rates = agent.get_firing_rates()
    return rates, pops


def _generate_reward_schedule(n_trials, reward_probs):
    """
    Generate probabilistic reward schedule.
    
    Args:
        n_trials: number of trials
        reward_probs: list of reward probabilities per action
        
    Returns:
        schedule: list of dicts, schedule[trial][action] = reward value (+1 or -1)
    """
    schedule = []
    for t in range(n_trials):
        trial_rewards = {}
        for a_idx, prob in enumerate(reward_probs):
            if np.random.rand() < prob:
                trial_rewards[a_idx] = 1.0
            else:
                trial_rewards[a_idx] = -1.0
        schedule.append(trial_rewards)
    return schedule


def _print_firing_rates(agent, pops):
    """Print firing rates grouped by population name."""
    rates = agent.get_firing_rates()
    
    # Group by name
    name_rates = {}
    for i, p in enumerate(pops):
        name = p['name']
        if name not in name_rates:
            name_rates[name] = []
        name_rates[name].append(rates[i])
    
    for name in ['Cx', 'CxI', 'dSPN', 'iSPN', 'FSI', 'GPe', 'STN', 'GPi', 'Th']:
        if name in name_rates:
            avg = np.mean(name_rates[name])
            vals = ', '.join(f'{r:.1f}' for r in name_rates[name])
            print(f"    {name:5s}: {avg:6.1f} Hz  [{vals}]")


if __name__ == '__main__':
    run_baseline_test()
