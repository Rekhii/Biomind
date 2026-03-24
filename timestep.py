import numpy as np
from biomind.params import NMDA_ALPHA, REFRACTORY_STEPS


def timestep(a):
    """
    Advance the simulation by one dt step.
    
    This is the entire neural computation per timestep:
    1. Reset spike indicators
    2. Update external synaptic input (Ornstein-Uhlenbeck)
    3. Propagate spikes through lateral connections
    4. Compute T-current (rebound bursting for STN/GPe)
    5. Update membrane voltage
    6. Detect spikes
    7. Update dopamine plasticity
    8. Update firing rate buffer
    
    Args:
        a: Agent object with all state arrays
    """
    n_pops = a.n_pops
    dt = a.dt
    newspikes = [[] for _ in range(n_pops)]
    

    # STEP 1: Reset pre/post spike indicators
    for i in range(n_pops):
        a.dpmn_XPRE[i] *= 0
        a.dpmn_XPOST[i] *= 0
    

    # STEP 2: External input (Ornstein-Uhlenbeck process per receptor)
    for i in range(n_pops):
        # AMPA external
        a.ExtMuS_AMPA[i] = (a.MeanExtEff_AMPA[i] * a.FreqExt_AMPA[i] 
                            * 0.001 * a.MeanExtCon_AMPA[i] * a.Tau_AMPA[i])
        a.ExtSigmaS_AMPA[i] = (a.MeanExtEff_AMPA[i] 
                               * np.sqrt(a.Tau_AMPA[i] * 0.5 * a.FreqExt_AMPA[i] 
                                         * 0.001 * a.MeanExtCon_AMPA[i]))
        a.ExtS_AMPA[i] += (dt / a.Tau_AMPA[i] * (-a.ExtS_AMPA[i] + a.ExtMuS_AMPA[i])
                          + a.ExtSigmaS_AMPA[i] * np.sqrt(dt * 2.0 / a.Tau_AMPA[i])
                          * np.random.normal(size=len(a.Tau_AMPA[i])))
        a.LS_AMPA[i] *= np.exp(-dt / a.Tau_AMPA[i])
        
        # GABA external
        a.ExtMuS_GABA[i] = (a.MeanExtEff_GABA[i] * a.FreqExt_GABA[i] 
                            * 0.001 * a.MeanExtCon_GABA[i] * a.Tau_GABA[i])
        a.ExtSigmaS_GABA[i] = (a.MeanExtEff_GABA[i] 
                               * np.sqrt(a.Tau_GABA[i] * 0.5 * a.FreqExt_GABA[i] 
                                         * 0.001 * a.MeanExtCon_GABA[i]))
        a.ExtS_GABA[i] += (dt / a.Tau_GABA[i] * (-a.ExtS_GABA[i] + a.ExtMuS_GABA[i])
                          + a.ExtSigmaS_GABA[i] * np.sqrt(dt * 2.0 / a.Tau_GABA[i])
                          * np.random.normal(size=len(a.Tau_AMPA[i])))
        a.LS_GABA[i] *= np.exp(-dt / a.Tau_GABA[i])
        
        # NMDA external
        a.ExtMuS_NMDA[i] = (a.MeanExtEff_NMDA[i] * a.FreqExt_NMDA[i] 
                            * 0.001 * a.MeanExtCon_NMDA[i] * a.Tau_NMDA[i])
        a.ExtSigmaS_NMDA[i] = (a.MeanExtEff_NMDA[i] 
                               * np.sqrt(a.Tau_NMDA[i] * 0.5 * a.FreqExt_NMDA[i] 
                                         * 0.001 * a.MeanExtCon_NMDA[i]))
        a.ExtS_NMDA[i] += (dt / a.Tau_NMDA[i] * (-a.ExtS_NMDA[i] + a.ExtMuS_NMDA[i])
                          + a.ExtSigmaS_NMDA[i] * np.sqrt(dt * 2.0 / a.Tau_NMDA[i])
                          * np.random.normal(size=len(a.Tau_AMPA[i])))
        a.LS_NMDA[i] *= np.exp(-dt / a.Tau_NMDA[i])
        
        a.timesincelastspike[i] += dt
        a.Ptimesincelastspike[i] += dt
    

    # STEP 3: Spike propagation through lateral connections
    # AMPA
    for src in range(n_pops):
        for dest in range(n_pops):
            if a.AMPA_con[src][dest] is not None:
                for neuron in a.spikes[src]:
                    a.LS_AMPA[dest] += (a.AMPA_eff[src][dest][neuron] 
                                        * a.AMPA_con[src][dest][neuron])
                    # Track pre-synaptic cortical spikes for plasticity
                    a.dpmn_XPRE[dest] = np.maximum(
                        a.dpmn_XPRE[dest],
                        a.dpmn_cortex[src][neuron] 
                        * a.AMPA_con[src][dest][neuron]
                        * np.sign(a.dpmn_type[dest])
                    )
    
    # GABA
    for src in range(n_pops):
        for dest in range(n_pops):
            if a.GABA_con[src][dest] is not None:
                for neuron in a.spikes[src]:
                    a.LS_GABA[dest] += (a.GABA_eff[src][dest][neuron] 
                                        * a.GABA_con[src][dest][neuron])
    
    # NMDA (with saturation)
    for src in range(n_pops):
        for dest in range(n_pops):
            if a.NMDA_con[src][dest] is not None:
                for neuron in a.spikes[src]:
                    # Decay last conductance
                    a.LastConductanceNMDA[src][dest][neuron] *= np.exp(
                        -a.Ptimesincelastspike[src][neuron] / a.Tau_NMDA[dest]
                    )
                    # Add new conductance with saturation
                    a.LS_NMDA[dest] += (
                        a.NMDA_eff[src][dest][neuron] 
                        * a.NMDA_con[src][dest][neuron] 
                        * NMDA_ALPHA 
                        * (1.0 - a.LastConductanceNMDA[src][dest][neuron])
                    )
                    # Update last conductance
                    a.LastConductanceNMDA[src][dest][neuron] += (
                        NMDA_ALPHA * (1.0 - a.LastConductanceNMDA[src][dest][neuron])
                    )
    

    # STEP 4: T-current (rebound bursting for STN and GPe)
    for i in range(n_pops):
        cond = (a.V[i] < a.V_h[i]).astype(int)
        # De-inactivation when hyperpolarized
        a.h[i] += cond * dt * (1.0 - a.h[i]) / a.tauhp[i]
        # Inactivation when depolarized
        a.h[i] += (1 - cond) * dt * (-a.h[i]) / a.tauhm[i]
        # Rebound conductance (only active when depolarized AND h > 0)
        a.g_rb[i] = a.g_T[i] * a.h[i] * (1 - cond)
    

    # STEP 5: Membrane voltage update
    for i in range(n_pops):
        # Check threshold and refractory state
        cond = (a.V[i] <= a.Threshold[i]).astype(int)
        a.V[i] -= (a.V[i] - a.ResetPot[i]) * (1 - cond)  # reset if above threshold
        
        cond = cond * (a.RefrState[i] == 0).astype(int)
        a.RefrState[i] -= np.sign(a.RefrState[i]) * (1 - cond)  # decrement refractory
        
        # Anomalous delayed rectifier
        a.g_adr[i] = a.g_adr_max[i] / (
            1.0 + np.exp((a.V[i] - a.Vadr_h[i]) / a.Vadr_s[i])
        )
        
        # Outward rectifying K+
        a.dv[i] = a.V[i] + 55.0
        a.tau_n[i] = a.tau_k_max[i] / (
            np.exp(-a.dv[i] / 30.0) + np.exp(a.dv[i] / 30.0)
        )
        a.n_inif[i] = 1.0 / (1.0 + np.exp(-(a.V[i] - a.Vk_h[i]) / a.Vk_s[i]))
        a.n_k[i] += cond * (-dt / a.tau_n[i]) * (a.n_k[i] - a.n_inif[i])
        a.g_k[i] = a.g_k_max[i] * a.n_k[i]
        
        # Main voltage equation: leak + AHP + ADR + K + T-current
        a.V[i] += cond * (-dt) * (
            (a.V[i] - a.RestPot[i]) / a.Taum[i]
            + a.Ca[i] * a.g_ahp[i] / a.C[i] * 0.001 * (a.V[i] - a.Vk[i])
            + a.g_adr[i] / a.C[i] * (a.V[i] - a.ADRRevPot[i])
            + a.g_k[i] / a.C[i] * (a.V[i] - a.ADRRevPot[i])
            + a.g_rb[i] / a.C[i] * (a.V[i] - a.V_T[i])
        )
        
        # Calcium decay
        a.Ca[i] -= cond * a.Ca[i] * dt / a.Tau_ca[i]
        
        # Clamp auxiliary voltage for synaptic drive
        a.Vaux[i] = np.minimum(a.V[i], a.Threshold[i])
        
        # Synaptic currents
        # NMDA (with Mg2+ block)
        a.V[i] += cond * dt * (
            (a.RevPot_NMDA[i] - a.Vaux[i]) * 0.001 
            * (a.LS_NMDA[i] + a.ExtS_NMDA[i]) / a.C[i]
            / (1.0 + np.exp(-0.062 * a.Vaux[i] / 3.57))
        )
        
        # AMPA
        a.V[i] += cond * dt * (
            (a.RevPot_AMPA[i] - a.Vaux[i]) * 0.001 
            * (a.LS_AMPA[i] + a.ExtS_AMPA[i]) / a.C[i]
        )
        
        # GABA
        a.V[i] += cond * dt * (
            (a.RevPot_GABA[i] - a.Vaux[i]) * 0.001 
            * (a.LS_GABA[i] + a.ExtS_GABA[i]) / a.C[i]
        )
        
        # Optogenetic (kept for compatibility, usually zero)
        a.V[i] += cond * dt * (
            (a.RevPot_ChR2[i] - a.Vaux[i]) * (a.ExtS_Opto[i] > 0) 
            * 0.001 * a.ExtS_Opto[i] / a.C[i]
        )
        a.V[i] += cond * dt * (
            (a.RevPot_NpHR[i] - a.Vaux[i]) * (a.ExtS_Opto[i] < 0) 
            * 0.001 * (-a.ExtS_Opto[i]) / a.C[i]
        )
    

    # STEP 6: Spike detection and reset
    for i in range(n_pops):
        spiked = np.nonzero(a.V[i] > a.Threshold[i])[0]
        newspikes[i] = list(spiked)
        for neuron in spiked:
            a.V[i][neuron] = 0.0  # spike marker
            a.Ca[i][neuron] += a.alpha_ca[i][neuron]
            a.RefrState[i][neuron] = REFRACTORY_STEPS
            a.Ptimesincelastspike[i][neuron] = a.timesincelastspike[i][neuron]
            a.timesincelastspike[i][neuron] = 0.0
            a.dpmn_XPOST[i][neuron] = 1.0
    
    a.spikes = newspikes
    

    # STEP 7: Dopamine-modulated plasticity (3-factor learning)
    for i in range(n_pops):
        if a.dpmn_type[i][0] > 0:  # only for dSPN and iSPN
            # Dopamine decay
            a.dpmn_DAp[i] -= dt * a.dpmn_DAp[i] / a.dpmn_tauDOP[i]
            
            # Pre/post synaptic traces
            a.dpmn_APRE[i] += dt * (
                a.dpmn_dPRE[i] * a.dpmn_XPRE[i] - a.dpmn_APRE[i]
            ) / a.dpmn_tauPRE[i]
            
            a.dpmn_APOST[i] += dt * (
                a.dpmn_dPOST[i] * a.dpmn_XPOST[i] - a.dpmn_APOST[i]
            ) / a.dpmn_tauPOST[i]
            
            # Eligibility trace
            a.dpmn_E[i] += dt * (
                a.dpmn_XPOST[i] * a.dpmn_APRE[i] 
                - a.dpmn_XPRE[i] * a.dpmn_APOST[i] 
                - a.dpmn_E[i]
            ) / a.dpmn_tauE[i]
            
            # Total dopamine
            DA = a.dpmn_m[i] * (a.dpmn_DAp[i] + a.dpmn_DAt[i])
            
            # f(DA) nonlinearity
            if a.dpmn_type[i][0] < 1.5:  # D1
                fDA = _get_fDA_D1(DA, a.dpmn_x_fda[i], a.dpmn_y_fda[i])
                a.dpmn_fDA_D1[i] = fDA
            else:  # D2
                fDA = _get_fDA_D2(DA, a.dpmn_x_fda[i], a.dpmn_y_fda[i], 
                                  a.dpmn_d2_DA_eps[i])
                a.dpmn_fDA_D2[i] = fDA
            
            # Weight update: only cortical AMPA connections
            for src in range(n_pops):
                if a.dpmn_cortex[src][0] > 0:
                    if a.AMPA_con[src][i] is not None:
                        update = (dt * a.AMPA_con[src][i] 
                                 * a.dpmn_alphaw[i] * fDA * a.dpmn_E[i])
                        
                        # Clip update magnitude
                        update = np.clip(update, -1.0, 1.0)
                        
                        # Soft bounds: multiplicative scaling
                        pos_mask = (update > 0).astype(float)
                        neg_mask = (update < 0).astype(float)
                        
                        a.AMPA_eff[src][i] += (
                            update * pos_mask * (a.dpmn_wmax[i] - a.AMPA_eff[src][i])
                        )
                        a.AMPA_eff[src][i] += (
                            update * neg_mask * (a.AMPA_eff[src][i] - 0.001)
                        )
    

    # STEP 8: Update firing rate buffer
    for i in range(n_pops):
        a.rollingbuffer[i][a.bufferpointer] = len(a.spikes[i])
    a.bufferpointer += 1
    if a.bufferpointer >= a.bufferlength:
        a.bufferpointer = 0


def multi_timestep(a, n_steps):
    """Run multiple timesteps."""
    for _ in range(n_steps):
        timestep(a)


def _get_fDA_D1(DA, x_fda, y_fda):
    """
    f(DA) for D1 neurons.
    Linear with slope y/x for DA > -x, saturates at -y below.
    Positive DA -> positive fDA -> LTP (with positive alphaw).
    """
    fda = np.zeros(len(DA))
    below = DA < -x_fda
    fda[below] = -y_fda[below]
    fda[~below] = (y_fda[~below] / x_fda[~below]) * DA[~below]
    return fda


def _get_fDA_D2(DA, x_fda, y_fda, eps):
    """
    f(DA) for D2 neurons.
    Linear with slope y/x*eps for DA < x, saturates at y*eps above.
    Positive DA -> positive fDA, but alphaw is NEGATIVE -> LTD on reward.
    eps scales down D2 sensitivity relative to D1.
    """
    fda = np.zeros(len(DA))
    above = DA > x_fda
    fda[above] = y_fda[above] * eps[above]
    fda[~above] = (y_fda[~above] / x_fda[~above]) * DA[~above] * eps[~above]
    return fda
