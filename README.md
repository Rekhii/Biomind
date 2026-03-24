# BioMind-BG: Biologically Faithful Basal Ganglia Simulation

A spiking neural network model of the cortico-basal-ganglia-thalamic (CBGT) circuit implemented in pure Python/NumPy.

## What It Does

Simulates ~3,000 leaky integrate-and-fire neurons organized into 9 brain nuclei, connected by 27 biologically constrained pathways across three receptor types (AMPA, GABA-A, NMDA). The network makes decisions through thalamic threshold crossing and learns from reward through dopamine-modulated three-factor synaptic plasticity.

## Key Features

- **Biologically validated**: All 9 nuclei fire at rates matching primate electrophysiology
- **Dopamine learning**: Three-factor rule (STDP + eligibility traces + D1/D2 dopamine)
- **Clinical accuracy**: Reproduces Parkinson's, Huntington's, and DBS effects through lesion studies
- **Pure Python/NumPy**: No external frameworks. ~900 lines of transparent, documented code
- **Zero dependencies**: Only requires Python 3.8+ and NumPy

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/biomind-bg.git
cd biomind-bg

# Run baseline test (checks firing rates)
python -m biomind.run

# Run full simulation with learning
python -c "from biomind.run import run_simulation; run_simulation(n_trials=5)"

# Run all experiments
python -m biomind.experiments
```

## Results

### Baseline Firing Rates

| Population | Model (Hz) | Expected (Hz) | Status |
|---|---|---|---|
| Cortex | 0.0 | 0-5 | PASS |
| D1-MSN (dSPN) | 3.8 | 0-8 | PASS |
| D2-MSN (iSPN) | 4.2 | 0-8 | PASS |
| FSI | 7.8 | 2-30 | PASS |
| GPe | 60.7 | 20-80 | PASS |
| STN | 24.9 | 8-40 | PASS |
| GPi | 71.0 | 30-100 | PASS |
| Thalamus | 6.8 | 0-20 | PASS |

### Lesion Studies

| Condition | GPi (Hz) | Thalamus (Hz) | Clinical Match |
|---|---|---|---|
| Healthy | 71.0 | 6.8 | Normal |
| Parkinson (D1 loss) | 81.0 | 3.4 | Bradykinesia |
| Huntington (iSPN loss) | 50.4 | 15.0 | Hyperkinesia |
| STN lesion (DBS) | 0.0 | 46.2 | Therapeutic effect |

## Architecture

```
biomind/
    __init__.py          - Package init
    params.py            - All biological constants with units
    populations.py       - Population construction + connectivity matrices
    agent.py             - Neural state initialization
    timestep.py          - Core simulation (all differential equations)
    qlearning.py         - Q-values and RPE-to-dopamine conversion
    trial.py             - Trial state machine
    run.py               - Main simulation runner
    experiments.py       - All 5 paper experiments
    test_validation.py   - Validation test suite
```

## The Circuit

```
        Cortex (Cx)
       /    |    \
      v     v     v
   dSPN   iSPN   FSI -----> dSPN, iSPN (inhibition)
     |      |
     v      v
    GPi    GPe <---> STN
     |              /
     v             v
  Thalamus <----- GPi
     |
     v
   Cortex (feedback)
```

**Direct pathway (Go):** Cx -> dSPN --| GPi --| Thalamus (released)
**Indirect pathway (NoGo):** Cx -> iSPN --| GPe --| STN -> GPi -> Thalamus (suppressed)

## Part of BioMind

BioMind-BG is the first component of the BioMind project, which aims to build a biologically grounded conscious intelligence. The complete architecture comprises:

**System 1 (Brain):** Basal Ganglia, Thalamus, Cortical Columns, Prefrontal Cortex, Hippocampus, Global Workspace

**System 2 (Self-Improving):** Architecture Inspector, Performance Monitor, Algorithm Inventor, Self-Modification Sandbox, Knowledge Compiler

## Citation

If you use BioMind-BG in your research, please cite:

```
@article{rekhi2026biomind,
  title={BioMind-BG: A Biologically Faithful Spiking Model of the 
         Cortico-Basal-Ganglia-Thalamic Circuit for Action Selection 
         and Dopamine-Driven Reward Learning},
  author={Rekhi},
  year={2026},
  journal={arXiv preprint}
}
```

## Acknowledgments

Built upon the CBGTPy framework by the CoAxLab at Georgia State University. All parameters derived from the biophysical literature and CBGTPy's validated implementation.

## License

MIT License
