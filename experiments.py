import numpy as np
import json
import time
from biomind.params import TRIAL_DEFAULTS
from biomind.populations import build_population_data, build_connectivity
from biomind.agent import Agent
from biomind.timestep import multi_timestep
from biomind.qlearning import QLearner
from biomind.trial import TrialManager


def experiment_1_baseline():
    """Experiment 1: Baseline firing rates (no stimulus)."""
    print("\n" + "x" * 60)
    print("EXPERIMENT 1: Baseline Firing Rates")
    print("x" * 60)
    
    np.random.seed(42)
    pops, pop_index = build_population_data(2)
    connectivity = build_connectivity(pops, pop_index, 2)
    agent = Agent(pops, connectivity)
    
    for ch in range(2):
        popid = pop_index[('Cx', ch)]
        agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
    
    multi_timestep(agent, 5000)
    
    rates = agent.get_firing_rates()
    name_rates = {}
    for i, p in enumerate(pops):
        name = p['name']
        if name not in name_rates:
            name_rates[name] = []
        name_rates[name].append(rates[i])
    
    results = {}
    for name in ['Cx', 'CxI', 'dSPN', 'iSPN', 'FSI', 'GPe', 'STN', 'GPi', 'Th']:
        avg = np.mean(name_rates[name])
        results[name] = round(avg, 1)
        print(f"  {name:5s}: {avg:6.1f} Hz")
    
    return results


def experiment_2_learning(n_trials=20, seed=42):
    """Experiment 2: 2-choice reward learning (80/20 split)."""
    print("\n" + "x" * 60)
    print(f"EXPERIMENT 2: Reward Learning ({n_trials} trials)")
    print("x" * 60)
    
    np.random.seed(seed)
    pops, pop_index = build_population_data(2)
    connectivity = build_connectivity(pops, pop_index, 2)
    agent = Agent(pops, connectivity)
    
    for ch in range(2):
        popid = pop_index[('Cx', ch)]
        agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
    
    multi_timestep(agent, 5000)
    
    reward_probs = [0.8, 0.2]
    reward_schedule = []
    for t in range(n_trials):
        trial_rewards = {}
        for a_idx, prob in enumerate(reward_probs):
            trial_rewards[a_idx] = 1.0 if np.random.rand() < prob else -1.0
        reward_schedule.append(trial_rewards)
    
    qlearner = QLearner(2)
    trial_mgr = TrialManager(agent, pop_index, 2,
                             reward_schedule=reward_schedule,
                             qlearner=qlearner)
    
    t0 = time.time()
    while trial_mgr.trial_num < n_trials:
        trial_complete = trial_mgr.step()
        if trial_complete:
            r = trial_mgr.results[-1]
            print(f"  Trial {r['trial']:3d}: action={r['chosen_action']}, "
                  f"reward={r['reward']:+.1f}, Q={r['Q_values']}")
    
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")
    
    choices = [r['chosen_action'] for r in trial_mgr.results]
    rewards = [r['reward'] for r in trial_mgr.results]
    
    # Compute choice proportions in blocks of 5
    blocks = []
    for i in range(0, len(choices), 5):
        block = choices[i:i+5]
        prop_0 = sum(1 for c in block if c == 0) / len(block)
        blocks.append(prop_0)
    
    print(f"\n  Action 0 proportion by block of 5: {[round(b, 2) for b in blocks]}")
    print(f"  Final Q-values: {qlearner.Q}")
    print(f"  Total reward: {sum(rewards):.0f}")
    
    return {
        'choices': [int(c) for c in choices],
        'rewards': [float(r) for r in rewards],
        'Q_history': [q.tolist() for q in qlearner.Q_history],
        'DA_history': [float(d) for d in qlearner.DA_history],
        'blocks': [round(b, 2) for b in blocks],
    }


def experiment_3_reversal(n_trials_per_phase=10, seed=42):
    """Experiment 3: Reward reversal. Phase 1: 80/20. Phase 2: 20/80."""
    print("\n" + "x" * 60)
    print(f"EXPERIMENT 3: Reward Reversal ({n_trials_per_phase} + {n_trials_per_phase} trials)")
    print("x" * 60)
    
    n_trials = n_trials_per_phase * 2
    np.random.seed(seed)
    pops, pop_index = build_population_data(2)
    connectivity = build_connectivity(pops, pop_index, 2)
    agent = Agent(pops, connectivity)
    
    for ch in range(2):
        popid = pop_index[('Cx', ch)]
        agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
    
    multi_timestep(agent, 5000)
    
    # Phase 1: action 0 = 80%, action 1 = 20%
    # Phase 2: action 0 = 20%, action 1 = 80% (REVERSED)
    reward_schedule = []
    for t in range(n_trials):
        trial_rewards = {}
        if t < n_trials_per_phase:
            probs = [0.8, 0.2]
        else:
            probs = [0.2, 0.8]  # REVERSED
        for a_idx, prob in enumerate(probs):
            trial_rewards[a_idx] = 1.0 if np.random.rand() < prob else -1.0
        reward_schedule.append(trial_rewards)
    
    qlearner = QLearner(2)
    trial_mgr = TrialManager(agent, pop_index, 2,
                             reward_schedule=reward_schedule,
                             qlearner=qlearner)
    
    t0 = time.time()
    while trial_mgr.trial_num < n_trials:
        trial_complete = trial_mgr.step()
        if trial_complete:
            r = trial_mgr.results[-1]
            phase = "Phase1" if r['trial'] < n_trials_per_phase else "Phase2"
            print(f"  {phase} Trial {r['trial']:3d}: action={r['chosen_action']}, "
                  f"reward={r['reward']:+.1f}, Q={r['Q_values']}")
    
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")
    
    choices = [int(r['chosen_action']) for r in trial_mgr.results]
    
    phase1_choices = choices[:n_trials_per_phase]
    phase2_choices = choices[n_trials_per_phase:]
    
    p1_prop0 = sum(1 for c in phase1_choices if c == 0) / len(phase1_choices)
    p2_prop0 = sum(1 for c in phase2_choices if c == 0) / len(phase2_choices)
    
    print(f"\n  Phase 1 (80/20): Action 0 chosen {p1_prop0*100:.0f}%")
    print(f"  Phase 2 (20/80): Action 0 chosen {p2_prop0*100:.0f}%")
    print(f"  Reversal detected: {p2_prop0 < p1_prop0}")
    
    return {
        'choices': choices,
        'Q_history': [q.tolist() for q in qlearner.Q_history],
        'phase1_prop0': round(p1_prop0, 2),
        'phase2_prop0': round(p2_prop0, 2),
        'reversal_detected': p2_prop0 < p1_prop0,
    }


def experiment_4_lesions(seed=42):
    """Experiment 4: Lesion studies simulating Parkinson's and Huntington's."""
    print("\n" + "x" * 60)
    print("EXPERIMENT 4: Lesion Studies")
    print("x" * 60)
    
    conditions = {
        'healthy': {},
        'parkinson_dSPN': {'lesion_pop': 'dSPN', 'lesion_factor': 0.2},   # 80% D1 loss
        'parkinson_iSPN': {'lesion_pop': 'iSPN', 'lesion_factor': 5.0},   # D2 overactive
        'huntington': {'lesion_pop': 'iSPN', 'lesion_factor': 0.1},       # 90% iSPN loss
        'stn_lesion': {'lesion_pop': 'STN', 'lesion_factor': 0.1},        # STN DBS-like
    }
    
    all_results = {}
    
    for cond_name, cond_params in conditions.items():
        print(f"\n  --- {cond_name} ---")
        np.random.seed(seed)
        
        pops, pop_index = build_population_data(2)
        connectivity = build_connectivity(pops, pop_index, 2)
        agent = Agent(pops, connectivity)
        
        # Apply lesion: scale efficacy of all outputs from lesioned population
        if 'lesion_pop' in cond_params:
            lesion_name = cond_params['lesion_pop']
            factor = cond_params['lesion_factor']
            
            for ch in range(2):
                key = (lesion_name, ch)
                if key in pop_index:
                    src_idx = pop_index[key]
                    for dest_idx in range(len(pops)):
                        for receptor in ['AMPA', 'GABA', 'NMDA']:
                            eff = getattr(agent, f'{receptor}_eff')
                            if eff[src_idx][dest_idx] is not None:
                                eff[src_idx][dest_idx] *= factor
                                print(f"    Scaled {lesion_name}(ch{ch})->{pops[dest_idx]['name']} "
                                      f"{receptor} by {factor}")
        
        for ch in range(2):
            popid = pop_index[('Cx', ch)]
            agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
        
        multi_timestep(agent, 5000)
        
        rates = agent.get_firing_rates()
        name_rates = {}
        for i, p in enumerate(pops):
            name = p['name']
            if name not in name_rates:
                name_rates[name] = []
            name_rates[name].append(rates[i])
        
        result = {}
        for name in ['Cx', 'dSPN', 'iSPN', 'GPe', 'STN', 'GPi', 'Th']:
            avg = np.mean(name_rates[name])
            result[name] = round(avg, 1)
            print(f"    {name:5s}: {avg:6.1f} Hz")
        
        all_results[cond_name] = result
    
    # Print comparison table
    print(f"\n  {'Condition':<20s} {'GPi':>6s} {'Th':>6s} {'dSPN':>6s} {'iSPN':>6s} {'STN':>6s}")
    print("  " + "-" * 50)
    for cond_name, result in all_results.items():
        print(f"  {cond_name:<20s} {result['GPi']:6.1f} {result['Th']:6.1f} "
              f"{result['dSPN']:6.1f} {result['iSPN']:6.1f} {result['STN']:6.1f}")
    
    return all_results


def experiment_5_multichoice(seed=42):
    """Experiment 5: Scale to 3 and 4 action choices."""
    print("\n" + "x" * 60)
    print("EXPERIMENT 5: Multi-Choice Scaling")
    print("x" * 60)
    
    all_results = {}
    
    for n_actions in [2, 3, 4]:
        print(f"\n  --- {n_actions} actions ---")
        np.random.seed(seed)
        
        pops, pop_index = build_population_data(n_actions)
        connectivity = build_connectivity(pops, pop_index, n_actions)
        agent = Agent(pops, connectivity)
        
        print(f"    Populations: {len(pops)}")
        
        for ch in range(n_actions):
            popid = pop_index[('Cx', ch)]
            agent.FreqExt_AMPA[popid] = np.zeros(pops[popid]['N'])
        
        multi_timestep(agent, 5000)
        
        rates = agent.get_firing_rates()
        name_rates = {}
        for i, p in enumerate(pops):
            name = p['name']
            if name not in name_rates:
                name_rates[name] = []
            name_rates[name].append(rates[i])
        
        result = {}
        for name in ['GPi', 'Th', 'dSPN', 'iSPN', 'STN', 'GPe']:
            avg = np.mean(name_rates[name])
            result[name] = round(avg, 1)
            print(f"    {name:5s}: {avg:6.1f} Hz  (channels: {[round(r,1) for r in name_rates[name]]})")
        
        result['n_pops'] = len(pops)
        result['n_actions'] = n_actions
        all_results[f'{n_actions}_actions'] = result
    
    return all_results


if __name__ == '__main__':
    all_data = {}
    
    all_data['exp1'] = experiment_1_baseline()
    all_data['exp4'] = experiment_4_lesions()
    all_data['exp5'] = experiment_5_multichoice()
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print("\n\nResults saved to experiment_results.json")
    print("\nNOTE: Experiments 2 (learning) and 3 (reversal) require")
    print("longer runtime. Run separately with:")
    print("  python -c \"from biomind.experiments import experiment_2_learning; experiment_2_learning(20)\"")
    print("  python -c \"from biomind.experiments import experiment_3_reversal; experiment_3_reversal(10)\"")
