# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:22:01 2026

@author: Cvke
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numba import jit


def generate_phase_centers(m, max_wid, min_gap, rng):
    if (m - 1) * min_gap > max_wid:
        raise ValueError("Cannot generate a valid sequence because (m-1)*min_gap > max_wid")
    
    remaining_space = max_wid - (m - 1) * min_gap
    random_points = rng.uniform(low=0, high=1, size=m + 1)
    total = np.sum(random_points)
    normalized_gaps = random_points / total * remaining_space
    
    cluster_centers = []
    current = normalized_gaps[0]
    cluster_centers.append(current)
    for i in range(1, m):
        current += min_gap + normalized_gaps[i]
        cluster_centers.append(current)
    return cluster_centers

def distribute_multinomial(n, m, rng):
    remaining_apples = n - m
    if remaining_apples <= 0:
        return [1] * m
    # 使用 rng.multinomial
    allocations = rng.multinomial(remaining_apples, [1/m]*m)
    allocations = allocations + 1
    return allocations.tolist()

def generate_clustered_phases(n, m, max_wid, min_gap, sigma, rng):
    cluster_centers = generate_phase_centers(m, max_wid, min_gap, rng)
    points_in_cluster = distribute_multinomial(n, m, rng)
    
    phases_sim = []
    for center, count in zip(cluster_centers, points_in_cluster):
        generated_points = rng.normal(loc=center, scale=sigma, size=int(count))
        generated_points = generated_points % 1.0
        phases_sim.extend(generated_points)
    
    phases_sim = np.array(phases_sim)
    rng.shuffle(phases_sim) 
    return phases_sim, points_in_cluster, cluster_centers

def generate_burst_sequences(period_og, t_obs, n, m, max_wid, min_gap, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    phases_sim, points_in_cluster, cluster_centers = generate_clustered_phases(n, m, max_wid, min_gap, sigma, rng)
    
    start_s_i = phases_sim * period_og
    bursts_i = start_s_i + rng.random(len(phases_sim)) * t_obs
    adjusted_bursts_i = start_s_i + np.round((bursts_i - start_s_i) / period_og) * period_og
    bursts_sim = np.sort(np.array(adjusted_bursts_i))
    
    return bursts_sim, phases_sim


@jit(nopython=True, parallel=False) 
def compute_chi2_numba_1D(samples, period, nbins):
    N = len(samples)
    if N == 0: return 0.0
    expected_count = N / nbins 
    count_bins = np.zeros(nbins, dtype=np.int32)
    
    factor = 1.0 + (nbins * nbins - 1) / (6.0 * N * (nbins - 1))
    
    for n in range(N):
        phase = (samples[n] % period) / period
        bin_idx = int(phase * nbins)
        if bin_idx >= nbins:
            bin_idx = nbins - 1
        count_bins[bin_idx] += 1
    
    chi2_stat = 0.0
    for b in range(nbins):
        diff = count_bins[b] - expected_count
        chi2_stat += (diff * diff) / expected_count
    
    reduced_chi2 = (chi2_stat / (nbins - 1)) / factor
    return reduced_chi2

#%%

#%%
if __name__ == "__main__":
    
    # Visualize the mock TOA sequence and the corresponding phase distribution
    period_og = 1.707 # set period (s)
    t_obs = 7200 # T_obs (s)
    n = 25  # N_burst
    m = 1 # Emission sites
    max_wid = 1 # max window width
    min_gap = 0.1  # min gap
    sigma = 0.1  # Gaussian standard deviation


    bursts_sim, phases_sim = generate_burst_sequences(period_og, t_obs, n, m, max_wid, min_gap, sigma)
    bursts_sim_0s=(bursts_sim-bursts_sim[0])

    fig, ax=plt.subplots(1,1)
    plt.plot(bursts_sim_0s,range(len(bursts_sim_0s)),'o',alpha=0.8)
    plt.xlabel('Simulated TOA (s)')
    plt.ylabel('Counts')
    plt.text(0.05, 0.96, 
             '$T_{\\mathrm{obs(sim)}}$ = %i s\n$N_{\\mathrm{burst(sim)}}$ = %i\nperiod = %.3f s' % (t_obs,n,period_og), 
             transform=ax.transAxes, fontsize=20,
             verticalalignment='top', horizontalalignment='left')


    folded_phases = np.fmod(bursts_sim_0s/period_og,1.0)
    fig, ax=plt.subplots(1,1)
    plt.hist(folded_phases, np.linspace(0,1,21), weights=None, color='royalblue',alpha=1)
    plt.hist(folded_phases+1, np.linspace(1,2,21), weights=None,color='royalblue',alpha=1)
    plt.text(0.05, 0.96, 
             '$m$ = %i\n$\\sigma$ = %.1f' % (m,sigma), 
             transform=ax.transAxes, fontsize=20,
             verticalalignment='top', horizontalalignment='left')
    plt.ylabel('Counts')
    plt.xlabel('Phase of Simulated TOA')

    # Simulation
    n_values = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    m_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_trials = 10000 
    
    global_rng = np.random.default_rng(2026)
    
    chi2_matrix = np.zeros((len(n_values), len(m_values)))
    
    print("Starting simulation...")
    for i, n in enumerate(n_values):
        for j, m in enumerate(m_values):
            chi2_array = np.zeros(n_trials)
            for trial in range(n_trials):
                bursts_sim_0s, _ = generate_burst_sequences(
                    period_og, t_obs, n, m, 
                    max_wid=1.0, min_gap=0.1, sigma=0.1,
                    rng=global_rng
                )
                y = compute_chi2_numba_1D(bursts_sim_0s - bursts_sim_0s[0], period_og, 20)
                chi2_array[trial] = y
            
            chi2_matrix[i, j] = np.mean(chi2_array)
        print(f"Finished N_burst={n}")
    
    np.save(r"Data\chi2_matrix.npy", chi2_matrix)
    
