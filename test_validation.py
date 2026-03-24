import numpy as np
import sys


def test_baseline_firing_rates():
    """Test that baseline firing rates match biological data."""
    from biomind.run import run_baseline_test
    
    print("TEST 1: Baseline Firing Rates")
    print("-" * 40)
    
    rates, pops = run_baseline_test(seed=42, verbose=False)
    
    # Group rates by name
    name_rates = {}
    for i, p in enumerate(pops):
        name = p['name']
        if name not in name_rates:
            name_rates[name] = []
        name_rates[name].append(rates[i])
    
    # Expected ranges (Hz) from primate/rodent electrophysiology
    expected = {
        'Cx':   (0, 5),      # No stimulus = near silent
        'CxI':  (0, 10),
        'dSPN': (0, 8),      # Low spontaneous rate
        'iSPN': (0, 8),
        'FSI':  (2, 30),     # Active even at baseline
        'GPe':  (20, 80),    # Autonomous pacemaker
        'STN':  (8, 40),     # Tonic firing
        'GPi':  (30, 100),   # High tonic rate
        'Th':   (0, 20),     # Suppressed by GPi
    }
    
    all_pass = True
    for name, (lo, hi) in expected.items():
        if name in name_rates:
            avg = np.mean(name_rates[name])
            ok = lo <= avg <= hi
            status = "PASS" if ok else "FAIL"
            print(f"  {name:5s}: {avg:6.1f} Hz  (expected {lo}-{hi})  [{status}]")
            if not ok:
                all_pass = False
    
    return all_pass


def test_decision_making():
    """Test that the network can make decisions when stimulated."""
    from biomind.run import run_simulation
    
    print("\nTEST 2: Decision Making")
    print("-" * 40)
    
    results = run_simulation(n_trials=5, seed=42, plasticity=False, verbose=False)
    
    trial_results = results['trial_results']
    
    # Check that decisions were made (not all -1)
    choices = [r['chosen_action'] for r in trial_results]
    n_decisions = sum(1 for c in choices if c >= 0)
    
    print(f"  Decisions made: {n_decisions}/{len(choices)}")
    print(f"  Choices: {choices}")
    
    decision_pass = n_decisions >= 3  # at least 60% should result in decisions
    print(f"  [{('PASS' if decision_pass else 'FAIL')}]")
    
    return decision_pass


def test_dopamine_learning():
    """Test that dopamine-driven learning biases choices toward high-reward action."""
    from biomind.run import run_simulation
    
    print("\nTEST 3: Dopamine Learning")
    print("-" * 40)
    
    results = run_simulation(
        n_trials=10, seed=42, 
        reward_probs=[0.8, 0.2], 
        plasticity=True, 
        verbose=False
    )
    
    trial_results = results['trial_results']
    choices = [r['chosen_action'] for r in trial_results]
    
    # In the last 5 trials, action 0 should be chosen more often
    last_5 = choices[-5:]
    n_correct = sum(1 for c in last_5 if c == 0)
    
    print(f"  All choices: {choices}")
    print(f"  Last 5 trials: action 0 chosen {n_correct}/5 times")
    print(f"  Q-values: {results['qlearner'].Q}")
    
    # Q[0] should be higher than Q[1]
    q_correct = results['qlearner'].Q[0] > results['qlearner'].Q[1]
    print(f"  Q[0] > Q[1]: {q_correct}")
    
    learning_pass = q_correct
    print(f"  [{('PASS' if learning_pass else 'FAIL')}]")
    
    return learning_pass


def test_network_structure():
    """Test that the network has the correct number of populations and connections."""
    from biomind.populations import build_population_data, build_connectivity
    
    print("\nTEST 4: Network Structure")
    print("-" * 40)
    
    pops, pop_index = build_population_data(2)
    connectivity = build_connectivity(pops, pop_index, 2)
    
    # Should have 16 populations for 2 actions
    n_pops_correct = len(pops) == 16
    print(f"  Populations: {len(pops)} (expected 16)  [{'PASS' if n_pops_correct else 'FAIL'}]")
    
    # Check plastic connections exist (Cx -> dSPN AMPA, Cx -> iSPN AMPA)
    plastic = connectivity['AMPA'][2]  # plasticity matrix
    n_plastic = 0
    for i in range(len(pops)):
        for j in range(len(pops)):
            if plastic[i][j]:
                n_plastic += 1
    
    # Should have 4 plastic connections (Cx ch0->dSPN ch0, Cx ch1->dSPN ch1, same for iSPN)
    plastic_correct = n_plastic == 4
    print(f"  Plastic connections: {n_plastic} (expected 4)  [{'PASS' if plastic_correct else 'FAIL'}]")
    
    # Check key pathway exists: GPi -> Th GABA
    gaba_con = connectivity['GABA'][0]
    gpi_th_exists = False
    for ch in range(2):
        gpi_idx = pop_index[('GPi', ch)]
        th_idx = pop_index[('Th', ch)]
        if gaba_con[gpi_idx][th_idx] is not None:
            gpi_th_exists = True
    
    print(f"  GPi->Th GABA exists: {gpi_th_exists}  [{'PASS' if gpi_th_exists else 'FAIL'}]")
    
    return n_pops_correct and plastic_correct and gpi_th_exists


if __name__ == '__main__':
    print("=" * 60)
    print("BioMind-BG: Validation Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Baseline Firing Rates", test_baseline_firing_rates()))
    results.append(("Network Structure", test_network_structure()))
    results.append(("Decision Making", test_decision_making()))
    results.append(("Dopamine Learning", test_dopamine_learning()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {name:30s}  {'PASS' if passed else 'FAIL'}")
    
    all_passed = all(r[1] for r in results)
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
