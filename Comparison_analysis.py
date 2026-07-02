# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:56:38 2026

@author: Cvke
"""

import numpy as np
import tqdm
from scipy.stats import lognorm
from scipy.stats import expon
from numba import njit, prange, get_thread_id, get_num_threads
import time
import pandas as pd

# Data Reading
def Xreaddata(path, col_indices, new_col_names, startrow=0):
    if len(col_indices) != len(new_col_names):
        raise ValueError("The number of column indices must match the number of new column names.")
    df_0 = pd.read_csv(path, dtype=str)
    df = df_0.iloc[startrow:, col_indices]
    df.columns = new_col_names
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    df = df.fillna(np.nan)
    df = df.sort_values(by='t')
    df = df.reset_index(drop=True)
    return df_0, df
    
b_fast1=Xreaddata('Data\\20201124A\\Burst_Table\\FAST#1.csv',
            [0,2,6,4],['t','s','w','f'],
            startrow=0)
#%%
#Further reduce the data table
def Xselectdata2(df_s, nummin, dnum, sep):
    t_bins = np.arange(np.floor(df_s['t'].min()) - (1 - sep), np.ceil(df_s['t'].max()) + (1 - sep) + dnum, dnum)
    num, _ = np.histogram(df_s['t'], t_bins)
    date_s = np.where(num >= nummin)[0]
    date_s = date_s * dnum
    df_list = []
    for date in date_s:
        mask = np.logical_and(df_s['t'] > (date + np.floor(df_s['t'].min()) - (1 - sep)), 
                              df_s['t'] < (date + np.floor(df_s['t'].min()) - (1 - sep)) + dnum)
        df_temp = df_s.loc[mask, :].reset_index(drop=True)
        df_temp.columns = [f'{col}' for col in df_temp.columns] 
        df_list.append(df_temp)
    return df_s, df_list

def burstverify_anyday(bi,nummin,dnum,sep) :
    bi_ss=Xselectdata2(bi[1],nummin=nummin,dnum=dnum,sep=sep)
    b_s=[]    
    for i in range(len(bi_ss[1])):
        b_s.append(bi_ss[1][i]['t']*86400)
        bi_ss[1][i]['t_s']=b_s[i]   
        waitingtime=np.diff(b_s[i])
        bi_ss[1][i]['waitingtime']=np.concatenate(([np.inf],waitingtime))    
    return bi_ss
    
# Reduce FAST#1 data
b_ss_fast1=burstverify_anyday(b_fast1, nummin=1,dnum=1,sep=0)

mu_log = 4.5  
sigma_log = 1.4 

def simulate_toas_lognormal(targettoas, 
                              tolerance=0.1, 
                              min_waiting_time=0.4,
                              max_rejection_attempts=100,
                              rng=None):
    if rng is None:
        rng = np.random.default_rng()
    num_bursts=len(targettoas)
    target_duration=max(targettoas)-min(targettoas)
    for attempt in range(max_rejection_attempts):
        waiting_times = []
        toas = []
        current_toa = min(targettoas)    
        for i in range(num_bursts):
            sampled_wt = -1.0  
            while sampled_wt < min_waiting_time:
                sampled_wt = rng.lognormal(mean=mu_log, sigma=sigma_log)
            waiting_times.append(sampled_wt)
            current_toa += sampled_wt
            toas.append(current_toa)
        if target_duration is not None and tolerance is not None:
            simulated_duration = toas[-1]-toas[0] if toas else 0
            if abs(simulated_duration - target_duration) <= tolerance * target_duration:
                return toas, waiting_times
        else: 
            return toas, waiting_times
    return None, None



def simulate_toas_uniform(targettoas, 
                          min_waiting_time=0.4,
                          max_rejection_attempts=100,
                          rng=None):

    if rng is None:
        rng = np.random.default_rng()
    num_bursts = len(targettoas)
    t0 = min(targettoas)
    target_duration = max(targettoas) - t0  # T
    if min_waiting_time is not None and min_waiting_time > 0:
        if num_bursts * min_waiting_time > target_duration:
            return None, None
    for attempt in range(max_rejection_attempts):
        toas = np.sort(rng.uniform(low=min(targettoas), high=max(targettoas), size=num_bursts))
        waiting_times = np.diff(np.concatenate(([t0], toas)))
        if np.all(waiting_times >= min_waiting_time):
            return toas.tolist(), waiting_times.tolist()
    return None, None


#%%
import time
def generate_multiple_toa_sets(method,
                               num_sets_to_generate, 
                               targettoas, 
                               min_waiting_time_per_set,
                               max_rejection_attempts_per_set,
                               random_seed=None):
    rng = np.random.default_rng(random_seed)
    if method == 'lognormal' :
        tolerance_per_set=0.1
        all_toa_sets = []
        all_waiting_time_sets = []
        generated_count = 0
        total_attempts = 0 
        num_bursts_per_set = len(targettoas)
        target_duration_per_set = max(targettoas)-min(targettoas)
        print(f"Starting to generate {num_sets_to_generate} sets of lognormal TOA sequences...")
        print(f"Number of bursts={num_bursts_per_set}, Target duration={target_duration_per_set:.2f}s, Tolerance={tolerance_per_set*100:.0f}%")
        start_time = time.time()
    
        while generated_count < num_sets_to_generate:
            total_attempts += 1
            toas, waiting_times = simulate_toas_lognormal(
                targettoas,
                tolerance_per_set,
                min_waiting_time_per_set,
                max_rejection_attempts_per_set,
                rng=rng
            )
    
            if toas: 
                all_toa_sets.append(toas)
                if waiting_times: 
                     all_waiting_time_sets.append(waiting_times)
                generated_count += 1
                
            if total_attempts > 100000 and generated_count < num_sets_to_generate : 
                 print(f"\nWarning: Attempted {total_attempts} calls to simulate_toas_for_one_day,")
                 print(f"but only successfully generated {generated_count} sets of sequences.")
                 print("The parameter settings (number of bursts, target duration, tolerance) may be making rejection sampling very difficult.")
                 print("Please check the parameters or increase max_rejection_attempts_per_set.")
                 break
        end_time = time.time()
        total_time_taken = end_time - start_time
        print(f"\n--- Generation Complete ---")
        print(f"Successfully generated {generated_count} sets of lognormal TOA sequences.")
        print(f"Total time taken: {total_time_taken:.2f} seconds.")
        return np.array(all_toa_sets)
    
    elif method == 'uniform' :
        all_toa_sets = []
        all_waiting_time_sets = []
        generated_count = 0
        total_attempts = 0 
        num_bursts_per_set = len(targettoas)
        target_duration_per_set = max(targettoas)-min(targettoas)
        print(f"Starting to generate {num_sets_to_generate} sets of uniform TOA sequences...")
        print(f"Parameters: Number of bursts={num_bursts_per_set}, target duration={target_duration_per_set:.2f}s")
        start_time = time.time()
    
        while generated_count < num_sets_to_generate:
            total_attempts += 1
            toas, waiting_times = simulate_toas_uniform(
                targettoas,
                min_waiting_time_per_set,
                max_rejection_attempts_per_set,
                rng=rng
            )
    
            if toas: 
                all_toa_sets.append(toas)
                if waiting_times: 
                     all_waiting_time_sets.append(waiting_times)
                generated_count += 1

            if total_attempts > 100000 and generated_count < num_sets_to_generate : 
                 print(f"\nWarning: Attempted {total_attempts} calls to simulate_toas_for_one_day,")
                 print(f"but only successfully generated {generated_count} sets of sequences.")
                 print("The parameter settings (number of bursts, target duration, tolerance) may be making rejection sampling very difficult.")
                 print("Please check the parameters or increase max_rejection_attempts_per_set.")
                 break
                 
        end_time = time.time()
        total_time_taken = end_time - start_time
        print(f"\n--- Generation Complete ---")
        print(f"Successfully generated {generated_count} sets of uniform TOA sequences.")
        print(f"Total time taken: {total_time_taken:.2f} seconds.")
        return np.array(all_toa_sets)
    else:
        print('Wrong method') 


#%%
def get_cluster_representatives(df_day,
                                threshold,
                                mode):
    """
    df_day      : Single-day DataFrame, must contain columns 't_s', 't', 'waitingtime0',
    and the column brightness_col used to define brightness (e.g., 'fluence' or 'snr').
    threshold   : Clustering time threshold (seconds), waitingtime0 < threshold is considered the same cluster.
    mode        : 'first' / 'last' / 'mean' / 'brightest'
    Returns:
    df_rep : DataFrame after selecting one representative for each cluster.
    """
    df = df_day.sort_values('t_s').reset_index(drop=True).copy()
    t = df['t_s'].to_numpy(np.float64)
    dt_prev = np.empty_like(t)
    dt_prev[0] = np.inf
    dt_prev[1:] = np.diff(t)
    n = len(df)
    cluster_id = np.empty(n, dtype=np.int32)
    cid = 0
    for i in range(n):
        if i == 0:
            cluster_id[i] = cid
        else:
            if dt_prev[i] < threshold:
                cluster_id[i] = cid         
            else:
                cid += 1                    
                cluster_id[i] = cid
    df['cluster_id'] = cluster_id
    reps = []
    for cid, g in df.groupby('cluster_id', sort=False):
        if mode == 'first':
            rep = g.iloc[0].copy()
        elif mode == 'last':
            rep = g.iloc[-1].copy()
        elif mode == 'mean':
            rep = g.iloc[0].copy()
            t_s_mean = g['t_s'].mean()
            rep['t_s'] = t_s_mean
            rep['t'] = t_s_mean / 86400.0    # MJD
        elif mode == 'brightest':
            idx = g['s'].idxmax()
            rep = g.loc[idx].copy()
        else:
            raise ValueError("mode must be 'first', 'last', 'mean' or 'brightest'")
        reps.append(rep)
    df_rep = pd.DataFrame(reps).reset_index(drop=True)
    return df_rep

#%%
#%%
#%%

@njit(parallel=True,fastmath=True)
def phase_fold_and_calc_chi2_opt(t, p_arr, bins_num=30):
    N = t.shape[0]
    M = p_arr.shape[0]
    chi_arr = np.empty(M, dtype=np.float64)
    q = 1.0       # NO Williams correction
    E = N / bins_num                        
    inv_norm = 1.0 / (E * (bins_num - 1) * q) 
    for i in prange(M):
        p = p_arr[i]
        inv_p = 1.0 / p
        O = np.zeros(bins_num, dtype=np.int64)
        for n in range(N):
            phase = (t[n] * inv_p) % 1.0   # in [0, 1)
            idx = int(phase * bins_num)
            if idx == bins_num:
                idx = bins_num - 1
            O[idx] += 1
        s2 = 0.0
        for b in range(bins_num):
            cnt = float(O[b])
            s2 += cnt * cnt
        chi_arr[i] = (s2 - N * E) * inv_norm
    return chi_arr



def mctest_chi2_joint(pstart, pend, step, dayrange, num, threshold, mode, para, seed, null_hypothesis='poisson'):
    """
    Perform Monte Carlo simulation following the description in Gazith 2025, supporting uniform or lognormal distributions as the null hypothesis.
    
    """
    chi2_sim_days = []        # Maximum chi-square values for simulated data
    chi2_period_sim_days = [] # Corresponding periods for maximum chi-square values of simulated data
    chi2_days_obs = []        # Maximum chi-square values for observed data
    chi2_period_days_obs = [] # Corresponding periods for maximum chi-square values of observed data
    meanT_days = []           # Stores the mean timestamp for each set of simulated data
    p_arr = np.linspace(pstart, pend, int(np.round((pend - pstart) / step)) + 1, dtype=np.float64)
    
    for i in tqdm.tqdm(dayrange, desc=f'Processing (H0:{null_hypothesis})',leave=True):
        day = i
        qr_sim = np.zeros(num)
        period_sim = np.zeros(num)
        
        b_sss = get_cluster_representatives(b_ss_fast1[1][day], threshold=threshold, mode=mode)
        target_toas = np.array(b_sss['t_s'])
        

        mc_samples = generate_multiple_toa_sets(
            method=null_hypothesis,  
            num_sets_to_generate=num,
            targettoas=target_toas,
            min_waiting_time_per_set=threshold,
            max_rejection_attempts_per_set=1000,
            random_seed=seed + day
        )
        
        if mc_samples is None or len(mc_samples) == 0:
            print(f"Warning: Day {day} failed to generate samples.")
            continue

        meanT_days.append(np.mean(mc_samples, axis=1))
        

        rx_obs = phase_fold_and_calc_chi2_opt(target_toas, p_arr, bins_num=para)
        idx_max_obs = np.argmax(rx_obs)
        chi2_days_obs.append(rx_obs[idx_max_obs])
        chi2_period_days_obs.append(p_arr[idx_max_obs])
        
        for k in tqdm.tqdm(range(len(mc_samples)), desc=f'Day {i} {null_hypothesis} Samples', leave=True):
            rx_sim = phase_fold_and_calc_chi2_opt(mc_samples[k], p_arr, bins_num=para)
            idx_max_sim = np.argmax(rx_sim)
            qr_sim[k] = rx_sim[idx_max_sim]
            period_sim[k] = p_arr[idx_max_sim]
            
        chi2_sim_days.append(qr_sim)
        chi2_period_sim_days.append(period_sim)

    mc_results = {
        f'chi2_days_{null_hypothesis}': np.array(chi2_sim_days).astype(np.float32),
        f'chi2_period_days_{null_hypothesis}': np.array(chi2_period_sim_days),
        'chi2_days_obs': np.array(chi2_days_obs).astype(np.float32),
        'chi2_period_days_obs': np.array(chi2_period_days_obs),
        'meanT_days': np.array(meanT_days),
        'pstart': pstart,
        'pend': pend,
        'step': step,
        'dayrange': np.array(dayrange),
        'num': num,
        'threshold': threshold,
        'mode': mode,
        'bins_num': para,
        'seed': seed,
        'null_hypothesis': null_hypothesis
    }
    return mc_results

mc_results_chi2=mctest_chi2_joint(null_hypothesis='uniform',pstart=0.1,pend=10,step=1e-7,dayrange=[3,35],
                                  num=1, # if set to 500 the runtime would be ~ 1 hr
                                  threshold=0.4,mode='brightest', para=30, seed=1)
pf='MC Samples_Comparison\\'
filename=(f'chi2_null{mc_results_chi2['null_hypothesis']}_seed{mc_results_chi2['seed']}_b{mc_results_chi2['bins_num']}_num{mc_results_chi2['num']}'
         f'_ps{mc_results_chi2['pstart']}_pe{mc_results_chi2['pend']}_step{mc_results_chi2['step']}'
         f'_th{mc_results_chi2['threshold']}_{mc_results_chi2['mode']}.npz')
np.savez_compressed(pf+filename, **mc_results_chi2)

mc_results_chi2=mctest_chi2_joint(null_hypothesis='lognormal',pstart=0.1,pend=10,step=1e-7,dayrange=[3,35],
                                  num=1, # if set to 500 the runtime would be ~ 1 hr
                                  threshold=0.4,mode='brightest', para=30, seed=1)
pf='MC Samples_Comparison\\'
filename=(f'chi2_null{mc_results_chi2['null_hypothesis']}_seed{mc_results_chi2['seed']}_b{mc_results_chi2['bins_num']}_num{mc_results_chi2['num']}'
         f'_ps{mc_results_chi2['pstart']}_pe{mc_results_chi2['pend']}_step{mc_results_chi2['step']}'
         f'_th{mc_results_chi2['threshold']}_{mc_results_chi2['mode']}.npz')
np.savez_compressed(pf+filename, **mc_results_chi2)