# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:48:37 2026

@author: Cvke
"""

# Import necessary libraries and configure matplotlib parameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from numpy import inf
import math
import time
import tqdm
import gc
import os
import matplotlib as mpl
from scipy import stats
from numba import njit, prange, get_thread_id, get_num_threads
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mticker
from brokenaxes import brokenaxes
from pathlib import Path
from scipy.stats import norm
from astropy.time import Time
from astropy import units
from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris
from astropy.utils import iers
import json
mpl.rcdefaults()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 2.8
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.grid']='True'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = 'True'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.minor.visible'] ='True'
plt.rcParams['ytick.minor.visible'] ='True'
plt.rcParams['xtick.major.size'] =10
plt.rcParams['xtick.minor.size'] =5
plt.rcParams['xtick.major.width'] =2
plt.rcParams['xtick.minor.width'] =1
plt.rcParams['xtick.major.pad']=8
plt.rcParams['ytick.major.size'] =10
plt.rcParams['ytick.minor.size'] =5
plt.rcParams['ytick.major.width'] =2
plt.rcParams['ytick.minor.width'] =1
plt.rcParams['ytick.major.pad']=8
plt.rcParams['figure.subplot.wspace']=0.25
plt.rcParams['figure.subplot.hspace']=0.5
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.linewidth'] = 2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['savefig.bbox']='tight'
plt.rcParams['savefig.format']='pdf'
rc_dict = mpl.rcParams.copy()

#%%


def load_windows_txt(path_txt):
    with open(path_txt, 'r') as f:
        lines = f.read().splitlines()
    data_lines = [ln for ln in lines if (ln.strip() != "") and (not ln.strip().startswith('#'))]
    flat_numbers = ' '.join(data_lines).split()
    arr = np.array(flat_numbers, dtype=float).reshape(-1, 2)
    return arr




def Xreaddata(path, col_indices, new_col_names, startrow=0):
    if len(col_indices) != len(new_col_names):
        raise ValueError("Column indices length mismatch.")
    df_0 = pd.read_csv(path, dtype=str)
    df = df_0.iloc[startrow:, col_indices]
    df.columns = new_col_names
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    df = df.fillna(np.nan)
    df = df.sort_values(by='t').reset_index(drop=True)
    return df_0, df

def Xselectdata2(df_s, nummin, dnum, sep):
    t_min = np.floor(df_s['t'].min())
    t_max = np.ceil(df_s['t'].max())
    t_bins = np.arange(t_min - (1 - sep), t_max + (1 - sep) + dnum, dnum)
    num, _ = np.histogram(df_s['t'], t_bins)
    date_s = np.where(num >= nummin)[0] * dnum

    df_list = []
    base_t = np.floor(df_s['t'].min())
    for date in date_s:
        start_d = date + base_t - (1 - sep)
        end_d = start_d + dnum
        mask = (df_s['t'] > start_d) & (df_s['t'] < end_d)
        df_temp = df_s.loc[mask, :].reset_index(drop=True)
        df_list.append(df_temp)
    return df_s, df_list

def burstverify_anyday(bi, nummin, dnum, sep):
    bi_ss = Xselectdata2(bi[1], nummin=nummin, dnum=dnum, sep=sep)
    for i in range(len(bi_ss[1])):
        b_s = bi_ss[1][i]['t'].to_numpy(dtype=float) * 86400.0
        bi_ss[1][i]['t_s'] = b_s
        bi_ss[1][i]['waitingtime0'] = np.concatenate(([np.inf], np.diff(b_s)))
        bi_ss[1][i]['waitingtime1'] = np.concatenate((np.diff(b_s), [np.inf])) 
    return bi_ss

def get_cluster_representatives(df_day, threshold):
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
    for _, g in df.groupby('cluster_id', sort=False):
        idx = g['s'].idxmax()
        rep = g.loc[idx].copy()
        reps.append(rep)
    return pd.DataFrame(reps).reset_index(drop=True)



def count_events_in_windows_inclusive(event_times_mjd, windows_mjd_2col):
    event_times_mjd = np.asarray(event_times_mjd, dtype=float)
    starts = windows_mjd_2col[:, 0]
    ends = windows_mjd_2col[:, 1]
    
    counts = np.empty(len(starts), dtype=int)
    for i in range(len(starts)):
        counts[i] = np.sum((event_times_mjd >= starts[i]) & (event_times_mjd <= ends[i]))
    return counts

def prepare_day_windows_and_counts_with_win(day_df_any, windows_tcb_mjd):

    event_times_mjd = day_df_any['t'].to_numpy(float)
    counts_all = count_events_in_windows_inclusive(event_times_mjd, windows_tcb_mjd)
    idx = np.where(counts_all > 0)[0]
    if idx.size == 0:
        return None, None, None

    win_sel = windows_tcb_mjd[idx]
    counts_sel = counts_all[idx].astype(int)
    intervals_s = [(float(a) * 86400.0, float(b) * 86400.0) for a, b in win_sel]
    return intervals_s, counts_sel, win_sel



def generate_N_toaseries_with_windows_fixed_counts(N, intervals_s, counts_per_interval, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    counts = np.asarray(counts_per_interval, dtype=int)
    starts = np.array([a for a, b in intervals_s], dtype=np.float64)
    ends   = np.array([b for a, b in intervals_s], dtype=np.float64)

    order = np.argsort(starts)
    starts, ends, counts = starts[order], ends[order], counts[order]

    N_total = int(np.sum(counts))
    out = np.empty((N, N_total), dtype=np.float64)

    col = 0
    for a, b, Nj in zip(starts, ends, counts):
        Nj = int(Nj)
        if Nj == 0:
            continue
        seg = rng.uniform(a, b, size=(N, Nj)).astype(np.float64)
        seg.sort(axis=1)
        out[:, col:col+Nj] = seg
        col += Nj
    return out

def build_s_template_from_windows(day_df, win_sel):
    t = day_df['t'].to_numpy(float)
    s = day_df['s'].to_numpy(float)

    s_list = []
    for (a, b) in win_sel:
        m = (t >= a) & (t <= b)
        s_list.append(s[m])

    if len(s_list) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(s_list).astype(np.float64)




def make_perm_matrix_fy(N_mc, B, rng, dtype=np.int32):
    perm = np.tile(np.arange(B, dtype=dtype), (N_mc, 1))
    rows = np.arange(N_mc, dtype=np.int64)
    tmp = np.empty(N_mc, dtype=dtype)
    for i in range(B - 1, 0, -1):
        j = rng.integers(0, i + 1, size=N_mc, dtype=np.int64)
        tmp[:] = perm[:, i]
        perm[:, i] = perm[rows, j]
        perm[rows, j] = tmp
    return perm

@njit(parallel=True, fastmath=True)
def cluster_reps_brightest_given_perm(toas_matrix, perm, s_template, threshold):
    N_mc, B = toas_matrix.shape
    reps = np.empty((N_mc, B), dtype=np.float64)
    nrep = np.empty(N_mc, dtype=np.int32)
    for s in prange(N_mc):
        if B == 0:
            nrep[s] = 0
            continue
        out_i = 0
        best_idx = 0
        best_s = float(s_template[perm[s, 0]])
        for i in range(1, B + 1):
            end_cluster = False
            if i == B:
                end_cluster = True
            else:
                dt = toas_matrix[s, i] - toas_matrix[s, i - 1]
                if dt >= threshold:
                    end_cluster = True
            if not end_cluster:
                sv = float(s_template[perm[s, i]])
                if sv > best_s:
                    best_s = sv
                    best_idx = i
            else:
                reps[s, out_i] = toas_matrix[s, best_idx]
                out_i += 1
                if i < B:
                    best_idx = i
                    best_s = float(s_template[perm[s, i]])
        nrep[s] = out_i
        for k in range(out_i, B):
            reps[s, k] = 0.0
    return reps, nrep




@njit(parallel=True, fastmath=True)
def batch_find_max_chi2_fast_flex(toas_matrix, n_arr, p_arr, bins_num=30):
    N_sim, N_max = toas_matrix.shape
    M = p_arr.shape[0]
    n_threads = get_num_threads()

    bins_f = float(bins_num)
    bins_minus_1 = float(bins_num - 1)
    f_bins = bins_f

    inv_norm_s = np.zeros(N_sim, dtype=np.float64)
    nE_s = np.zeros(N_sim, dtype=np.float64)

    for s in range(N_sim):
        n = int(n_arr[s])
        if n >= 2:
            n_f = float(n)
            E = n_f / bins_f
            q = 1.0 + (bins_f + 1.0) / (6.0 * n_f)
            inv_norm_s[s] = 1.0 / (E * bins_minus_1 * q)
            nE_s[s] = n_f * E
        else:
            inv_norm_s[s] = 0.0
            nE_s[s] = 0.0

    tls_max_vals = np.full((n_threads, N_sim), -1.0, dtype=np.float64)
    tls_best_p   = np.zeros((n_threads, N_sim), dtype=np.float64)

    chunk_len = 4096
    n_chunks = (M + chunk_len - 1) // chunk_len
    UF = 4

    for c in prange(n_chunks):
        tid = get_thread_id()
        start_idx = c * chunk_len
        end_idx = min(start_idx + chunk_len, M)
        limit = start_idx + ((end_idx - start_idx) // UF) * UF

        multi_counts = np.zeros((UF, bins_num), dtype=np.int32)
        single_counts = np.zeros(bins_num, dtype=np.int32)

        for i in range(start_idx, limit, UF):
            f0, p0 = 1.0 / p_arr[i],   p_arr[i]
            f1, p1 = 1.0 / p_arr[i+1], p_arr[i+1]
            f2, p2 = 1.0 / p_arr[i+2], p_arr[i+2]
            f3, p3 = 1.0 / p_arr[i+3], p_arr[i+3]

            for s in range(N_sim):
                n = int(n_arr[s])
                if n < 2:
                    continue

                inv_norm = inv_norm_s[s]
                nE = nE_s[s]

                multi_counts[:] = 0
                for k in range(n):
                    val = toas_matrix[s, k]

                    ph = val * f0; ph -= np.floor(ph)
                    idx = int(ph * f_bins)
                    if idx >= bins_num: idx = bins_num - 1
                    multi_counts[0, idx] += 1

                    ph = val * f1; ph -= np.floor(ph)
                    idx = int(ph * f_bins)
                    if idx >= bins_num: idx = bins_num - 1
                    multi_counts[1, idx] += 1

                    ph = val * f2; ph -= np.floor(ph)
                    idx = int(ph * f_bins)
                    if idx >= bins_num: idx = bins_num - 1
                    multi_counts[2, idx] += 1

                    ph = val * f3; ph -= np.floor(ph)
                    idx = int(ph * f_bins)
                    if idx >= bins_num: idx = bins_num - 1
                    multi_counts[3, idx] += 1

                curr_max = tls_max_vals[tid, s]
                curr_best = tls_best_p[tid, s]

                s2 = 0.0
                for b in range(bins_num):
                    x = float(multi_counts[0, b]); s2 += x*x
                chi = (s2 - nE) * inv_norm
                if chi > curr_max:
                    curr_max = chi; curr_best = p0

                s2 = 0.0
                for b in range(bins_num):
                    x = float(multi_counts[1, b]); s2 += x*x
                chi = (s2 - nE) * inv_norm
                if chi > curr_max:
                    curr_max = chi; curr_best = p1

                s2 = 0.0
                for b in range(bins_num):
                    x = float(multi_counts[2, b]); s2 += x*x
                chi = (s2 - nE) * inv_norm
                if chi > curr_max:
                    curr_max = chi; curr_best = p2

                s2 = 0.0
                for b in range(bins_num):
                    x = float(multi_counts[3, b]); s2 += x*x
                chi = (s2 - nE) * inv_norm
                if chi > curr_max:
                    curr_max = chi; curr_best = p3

                tls_max_vals[tid, s] = curr_max
                tls_best_p[tid, s] = curr_best

        for i in range(limit, end_idx):
            f, p_val = 1.0 / p_arr[i], p_arr[i]
            for s in range(N_sim):
                n = int(n_arr[s])
                if n < 2:
                    continue

                inv_norm = inv_norm_s[s]
                nE = nE_s[s]

                single_counts[:] = 0
                for k in range(n):
                    val = toas_matrix[s, k]
                    ph = val * f; ph -= np.floor(ph)
                    idx = int(ph * f_bins)
                    if idx >= bins_num: idx = bins_num - 1
                    single_counts[idx] += 1

                s2 = 0.0
                for b in range(bins_num):
                    x = float(single_counts[b]); s2 += x*x
                chi = (s2 - nE) * inv_norm

                if chi > tls_max_vals[tid, s]:
                    tls_max_vals[tid, s] = chi
                    tls_best_p[tid, s] = p_val

    final_max_vals = np.empty(N_sim, dtype=np.float32)
    final_best_p = np.empty(N_sim, dtype=np.float64)

    for s in range(N_sim):
        g_max = -1.0
        g_p = 0.0
        for t in range(n_threads):
            if tls_max_vals[t, s] > g_max:
                g_max = tls_max_vals[t, s]
                g_p = tls_best_p[t, s]
        final_max_vals[s] = g_max
        final_best_p[s] = g_p

    return final_max_vals, final_best_p



def _simulate_one_day_chunked(
    *,
    day_idx: int,
    intervals_s,
    counts_sel: np.ndarray,
    s_template: np.ndarray,
    p_short: np.ndarray,
    p_long: np.ndarray,
    bins_num: int,
    cluster_threshold: float,
    N_pairs: int,
    sim_chunk_size: int,
    seed: int,
    chi_out: np.ndarray,
    P_out: np.ndarray,
    meanT_out: np.ndarray,
):


    rng_toa  = np.random.default_rng((seed, day_idx, 0))
    rng_perm = np.random.default_rng((seed, day_idx, 1))

    B = int(np.sum(counts_sel))
    if B != int(s_template.size):
        raise ValueError(f"Day{day_idx}: B(sum(counts))={B} != len(s_template)={s_template.size}")

    pbar = tqdm.tqdm(total=N_pairs, desc=f"Sims Unit{day_idx} (chunked)", leave=True)

    pos = 0
    while pos < N_pairs:
        n_blk = min(sim_chunk_size, N_pairs - pos)

        # 1) raw TOA block
        mc = generate_N_toaseries_with_windows_fixed_counts(n_blk, intervals_s, counts_sel, rng=rng_toa)
        meanT_out[pos:pos+n_blk] = np.mean(mc, axis=1)

        # 2) short 
        if p_short.size > 0:
            n_raw = np.full(n_blk, mc.shape[1], dtype=np.int32)
            chi_short, P_short = batch_find_max_chi2_fast_flex(mc, n_raw, p_short, bins_num)
        else:
            chi_short = np.full(n_blk, -1.0, dtype=np.float32)
            P_short   = np.zeros(n_blk, dtype=np.float64)

        # 3) long 
        if p_long.size > 0:
            perm = make_perm_matrix_fy(n_blk, mc.shape[1], rng_perm, dtype=np.int32)
            reps, nrep = cluster_reps_brightest_given_perm(mc, perm, s_template, float(cluster_threshold))
            del perm

            chi_long, P_long = batch_find_max_chi2_fast_flex(reps, nrep, p_long, bins_num)
            del reps, nrep
        else:
            chi_long = np.full(n_blk, -1.0, dtype=np.float32)
            P_long   = np.zeros(n_blk, dtype=np.float64)

        # 4) max over segments
        chi_blk = np.maximum(chi_short, chi_long).astype(np.float32)
        mask = chi_short >= chi_long
        P_blk = np.where(mask, P_short, P_long).astype(np.float64)

        chi_out[pos:pos+n_blk] = chi_blk
        P_out[pos:pos+n_blk]   = P_blk

        del mc, chi_short, chi_long, P_short, P_long, chi_blk, P_blk

        pos += n_blk
        pbar.update(n_blk)

    pbar.close()

#%%

#%%
# ============================================================
# A) Small helpers
# ============================================================

def p_to_sigma(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-300, 1 - 1e-16)
    out = norm.isf(p)
    return float(out) if out.ndim == 0 else out


def sigma_to_p(z):
    z = np.asarray(z, dtype=np.float64)
    out = norm.sf(z)
    return float(out) if out.ndim == 0 else out


def p_to_neglogp(p):
    """
    Convert probability p to the single-epoch significance score Q:
        Q = -log10(p)
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-300, 1.0)
    out = -np.log10(p)
    return float(out) if out.ndim == 0 else out


def neglogp_to_p(q):
    """
    Inverse transform:
        p = 10^{-Q}
    """
    q = np.asarray(q, dtype=np.float64)
    out = np.power(10.0, -q)
    return float(out) if out.ndim == 0 else out


def threshold_to_p(metric, thr):
    """
    Utility conversion among threshold metrics.
    """
    thr = np.asarray(thr, dtype=np.float64)
    if metric == "p":
        out = thr
    elif metric == "neglogp":
        out = neglogp_to_p(thr)
    elif metric == "z":
        out = sigma_to_p(thr)
    else:
        raise ValueError("metric must be one of {'p','neglogp','z'}")
    return out


def get_windows_for_unit(windows_obj, unit_idx):
    if isinstance(windows_obj, dict):
        return windows_obj[unit_idx]
    elif isinstance(windows_obj, (list, tuple)):
        return windows_obj[unit_idx]
    else:
        return windows_obj


def _all_pairs_indices(n_unit):
    return np.triu_indices(n_unit, k=1)


def _chi_to_empirical_p(chi_values, chi_sorted):
    """
    Convert a chi value to an empirical tail probability using
    the unit-specific null library:
        p = P(chi_sim >= chi_any)
    """
    M = chi_sorted.size
    left = np.searchsorted(chi_sorted, chi_values, side='left')
    nge = M - left
    p = (nge) / (M)
    return p


# ============================================================
# B) Observed strongest peak per unit
# ============================================================

def search_one_unit_observed(unit_df, p_short, p_long, bins_num=20, cluster_threshold=0.5):
    """
    Observed strongest peak for one analysis unit:
      - raw TOAs for short periods
      - brightest-cluster representatives for long periods
      - take the maximum over the two search segments
    """
    t_all = unit_df['t_s'].to_numpy(np.float64)
    rep_df = get_cluster_representatives(unit_df, threshold=cluster_threshold)
    t_rep = rep_df['t_s'].to_numpy(np.float64)

    def _max_obs(t, p_seg):
        if p_seg.size == 0 or t.size < 2:
            return -1.0, 0.0
        n = np.array([t.size], dtype=np.int32)
        chi, pp = batch_find_max_chi2_fast_flex(
            t.reshape(1, -1), n, p_seg, bins_num=bins_num
        )
        return float(chi[0]), float(pp[0])

    chi_short, P_short = _max_obs(t_all, p_short)
    chi_long,  P_long  = _max_obs(t_rep, p_long)

    if chi_short >= chi_long:
        chi_obs, P_obs = chi_short, P_short
    else:
        chi_obs, P_obs = chi_long, P_long

    out = {
        "chi_obs": float(chi_obs),
        "P_obs": float(P_obs),
        "N_raw": int(len(t_all)),
        "N_rep": int(len(t_rep)),
        "meanT_obs": float(np.mean(t_all)) if len(t_all) > 0 else np.nan,
    }
    return out


def build_unit_spec(
    unit_idx,
    unit_df,
    windows_obj,
    p_short,
    p_long,
    bins_num=20,
    cluster_threshold=0.5,
    extra_meta=None
):
    """
    Build one unit's specification for both observed search and null simulation.
    """
    windows_arr = get_windows_for_unit(windows_obj, unit_idx)

    intervals_s, counts_sel, win_sel = prepare_day_windows_and_counts_with_win(unit_df, windows_arr)
    if intervals_s is None:
        return None

    # enforce time-order consistency
    order = np.argsort(win_sel[:, 0])
    win_sel = win_sel[order]
    counts_sel = counts_sel[order]

    intervals_s = [(float(a) * 86400.0, float(b) * 86400.0) for a, b in win_sel]
    s_template = build_s_template_from_windows(unit_df, win_sel).astype(np.float64)

    n_event = len(unit_df)
    if int(np.sum(counts_sel)) != int(n_event):
        raise RuntimeError(
            f"Unit {unit_idx}: sum(counts_sel)={np.sum(counts_sel)} != n_event={n_event}. "
            "Likely a mismatch between burst times and window time system / boundaries."
        )

    obs = search_one_unit_observed(
        unit_df=unit_df,
        p_short=p_short,
        p_long=p_long,
        bins_num=bins_num,
        cluster_threshold=cluster_threshold
    )

    mjd_label = int(np.floor(np.nanmean(unit_df['t'].to_numpy(float))))

    spec = {
        "unit_idx": int(unit_idx),
        "mjd": int(mjd_label),
        "intervals_s": intervals_s,
        "counts_sel": counts_sel.astype(np.int32),
        "win_sel_mjd": win_sel.astype(np.float64),
        "s_template": s_template,
        "chi_obs": float(obs["chi_obs"]),
        "P_obs": float(obs["P_obs"]),
        "N_raw": int(obs["N_raw"]),
        "N_rep": int(obs["N_rep"]),
        "meanT_obs": float(obs["meanT_obs"]),
    }

    if extra_meta is not None:
        for k, v in extra_meta.items():
            spec[k] = v

    return spec


def build_all_unit_specs(
    unit_dfs,
    windows_obj,
    pstart,
    pend,
    step,
    bins_num=20,
    cluster_threshold=0.5,
    unit_meta_list=None
):
    p_arr = np.linspace(pstart, pend, int((pend - pstart) / step) + 1, dtype=np.float64)
    split = float(cluster_threshold) * 2.0
    p_short = p_arr[p_arr < split]
    p_long  = p_arr[p_arr >= split]

    specs = []
    for unit_idx, unit_df in enumerate(tqdm.tqdm(unit_dfs, desc="Build unit specs")):
        extra_meta = None if unit_meta_list is None else unit_meta_list[unit_idx]
        spec = build_unit_spec(
            unit_idx=unit_idx,
            unit_df=unit_df,
            windows_obj=windows_obj,
            p_short=p_short,
            p_long=p_long,
            bins_num=bins_num,
            cluster_threshold=cluster_threshold,
            extra_meta=extra_meta
        )
        if spec is not None:
            specs.append(spec)

    meta = {
        "p_start": float(pstart),
        "p_end": float(pend),
        "p_step": float(step),
        "bins_num": int(bins_num),
        "cluster_threshold": float(cluster_threshold),
        "split_period": float(split),
        "N_units": int(len(specs))
    }
    return specs, meta, p_short, p_long


# ============================================================
# C) Unit-library generation / save / load
# ============================================================

_UNIT_META_KEYS = ["dataset", "local_idx"]


def simulate_unit_library_from_spec(
    unit_spec,
    p_short,
    p_long,
    bins_num=20,
    cluster_threshold=0.5,
    N_mc=10000,
    sim_chunk_size=1000,
    seed=2026
):
    chi_sims = np.empty(N_mc, dtype=np.float32)
    P_sims   = np.empty(N_mc, dtype=np.float64)
    meanT_sims = np.empty(N_mc, dtype=np.float64)

    _simulate_one_day_chunked(
        day_idx=unit_spec["unit_idx"],
        intervals_s=unit_spec["intervals_s"],
        counts_sel=unit_spec["counts_sel"],
        s_template=unit_spec["s_template"],
        p_short=p_short,
        p_long=p_long,
        bins_num=bins_num,
        cluster_threshold=cluster_threshold,
        N_pairs=N_mc,
        sim_chunk_size=sim_chunk_size,
        seed=seed,
        chi_out=chi_sims,
        P_out=P_sims,
        meanT_out=meanT_sims,
    )

    lib = {
        "unit_idx": int(unit_spec["unit_idx"]),
        "mjd": int(unit_spec["mjd"]),
        "chi_obs": float(unit_spec["chi_obs"]),
        "P_obs": float(unit_spec["P_obs"]),
        "N_raw": int(unit_spec["N_raw"]),
        "N_rep": int(unit_spec["N_rep"]),
        "meanT_obs": float(unit_spec["meanT_obs"]),
        "chi_sims": chi_sims,
        "P_sims": P_sims,
        "meanT_sims": meanT_sims,
        "config": {
            "bins_num": int(bins_num),
            "cluster_threshold": float(cluster_threshold),
            "N_mc": int(N_mc),
            "sim_chunk_size": int(sim_chunk_size),
            "seed": int(seed),
        }
    }

    for k in _UNIT_META_KEYS:
        if k in unit_spec:
            lib[k] = unit_spec[k]

    return lib


def save_unit_library_npz(outdir, lib):
    os.makedirs(outdir, exist_ok=True)
    unit_idx = lib["unit_idx"]
    mjd = lib["mjd"]
    ds = lib["dataset"]
    baseseedx = lib["config"]["seed"] - unit_idx
    N_mcx = lib["config"]["N_mc"]

    fname = os.path.join(
        outdir,
        f"unitlib_seed{baseseedx}_N{N_mcx}_unit{unit_idx:03d}_{ds}_mjd{mjd}.npz"
    )

    meta = {
        "unit_idx": lib["unit_idx"],
        "mjd": lib["mjd"],
        "chi_obs": lib["chi_obs"],
        "P_obs": lib["P_obs"],
        "N_raw": lib["N_raw"],
        "N_rep": lib["N_rep"],
        "meanT_obs": lib["meanT_obs"],
        "config": lib["config"],
    }

    for k in _UNIT_META_KEYS:
        if k in lib:
            meta[k] = lib[k]

    np.savez_compressed(
        fname,
        chi_sims=lib["chi_sims"],
        P_sims=lib["P_sims"],
        meanT_sims=lib["meanT_sims"],
        meta_json=json.dumps(meta, ensure_ascii=False)
    )
    return fname


def load_unit_library_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))

    lib = {
        "unit_idx": int(meta["unit_idx"]),
        "mjd": int(meta["mjd"]),
        "chi_obs": float(meta["chi_obs"]),
        "P_obs": float(meta["P_obs"]),
        "N_raw": int(meta["N_raw"]),
        "N_rep": int(meta["N_rep"]),
        "meanT_obs": float(meta["meanT_obs"]),
        "config": meta["config"],
        "chi_sims": z["chi_sims"].astype(np.float32),
        "P_sims": z["P_sims"].astype(np.float64),
        "meanT_sims": z["meanT_sims"].astype(np.float64),
    }

    for k in _UNIT_META_KEYS:
        if k in meta:
            lib[k] = meta[k]

    return lib


def build_and_save_all_unit_libraries(
    specs,
    p_short,
    p_long,
    outdir,
    bins_num=20,
    cluster_threshold=0.5,
    N_mc=10000,
    sim_chunk_size=1000,
    base_seed=2026
):
    paths = []
    for k, spec in enumerate(tqdm.tqdm(specs, desc="Build all unit libraries", leave=True)):
        lib = simulate_unit_library_from_spec(
            unit_spec=spec,
            p_short=p_short,
            p_long=p_long,
            bins_num=bins_num,
            cluster_threshold=cluster_threshold,
            N_mc=N_mc,
            sim_chunk_size=sim_chunk_size,
            seed=base_seed + k
        )
        path = save_unit_library_npz(outdir, lib)
        paths.append(path)
    return paths


def load_all_unit_libraries(outdir, baseseedx, nlocal):
    files = sorted([
        os.path.join(outdir, x)
        for x in os.listdir(outdir)
        if x.endswith(".npz") and x.startswith(f"unitlib_seed{baseseedx}_N{nlocal}_")
    ])
    libs = [load_unit_library_npz(f) for f in files]
    libs = sorted(libs, key=lambda x: x["unit_idx"])
    return libs


# ============================================================
# D) Unit-level calibration: chi -> p / Q / z
# ============================================================

def calibrate_one_library(lib):
    chi_sims = lib["chi_sims"].astype(np.float64)
    chi_sorted = np.sort(chi_sims)

    p_obs = float(_chi_to_empirical_p(
        np.array([lib["chi_obs"]], dtype=np.float64), chi_sorted
    )[0])
    Q_obs = float(p_to_neglogp(p_obs))
    z_obs = float(p_to_sigma(p_obs))

    p_sims = _chi_to_empirical_p(chi_sims, chi_sorted).astype(np.float64)
    Q_sims = p_to_neglogp(p_sims).astype(np.float32)
    z_sims = p_to_sigma(p_sims).astype(np.float32)

    lib2 = dict(lib)
    lib2["p_obs"] = p_obs
    lib2["Q_obs"] = Q_obs
    lib2["z_obs"] = z_obs

    lib2["p_sims"] = p_sims
    lib2["Q_sims"] = Q_sims
    lib2["z_sims"] = z_sims
    return lib2


def calibrate_all_libraries(libs):
    libs2 = []
    for lib in libs:
        libs2.append(calibrate_one_library(lib))
    libs2 = sorted(libs2, key=lambda x: x["unit_idx"])
    return libs2


def build_observed_catalog_from_libraries(libs):
    rows = []
    for lib in libs:
        row = {
            "unit_idx": lib["unit_idx"],
            "mjd": lib["mjd"],
            "chi_obs": lib["chi_obs"],
            "P_obs": lib["P_obs"],
            "p_obs": lib["p_obs"],
            "Q_obs": lib["Q_obs"],
            "z_obs": lib["z_obs"],
            "N_raw": lib["N_raw"],
            "N_rep": lib["N_rep"],
            "meanT_obs": lib["meanT_obs"] / 86400,
        }
        for k in _UNIT_META_KEYS:
            if k in lib:
                row[k] = lib[k]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("unit_idx").reset_index(drop=True)
    return df


# ============================================================
# E) Pair-level statistics: R, S, T
# ============================================================

def pair_R_from_Q(
    Q1,
    Q2,
    Q_ref_strong,
    Q_ref_weak
):
    """
    Ordered pair-significance score:

        R_ab = min(max(Q_a,Q_b)/Q_obs,strong,
                   min(Q_a,Q_b)/Q_obs,weak)
    """
    Q_strong = np.maximum(Q1, Q2)
    Q_weak = np.minimum(Q1, Q2)

    R = np.minimum(
        Q_strong / Q_ref_strong,
        Q_weak / Q_ref_weak
    )

    return R.astype(np.float64)


def pair_S_from_periods(
    P_a,
    P_b,
    dP_ref,
    dP_floor=1e-5,
    cap_period_score=None
):
    """
    Period-closeness score:

        S_ab = dP_ref / max(|P_a - P_b|, dP_floor)

    Interpretation
    --------------
    S = 1:
        The pair is as close in period as the observed target pair.

    S > 1:
        The pair is closer in period than the observed target pair.

    S < 1:
        The pair is less close in period than the observed target pair.
    """
    P_a = np.asarray(P_a, dtype=np.float64)
    P_b = np.asarray(P_b, dtype=np.float64)

    dP = np.abs(P_a - P_b)
    S = dP_ref / np.maximum(dP, dP_floor)

    if cap_period_score is not None:
        S = np.minimum(S, float(cap_period_score))

    return S.astype(np.float64)


# ============================================================
# F) Global cache and sampling
# ============================================================

def prepare_global_fast_cache(libs):
    """
    Prepare stacked arrays for fast global assembly.
    """
    libs = sorted(libs, key=lambda x: x["unit_idx"])
    obs_df = build_observed_catalog_from_libraries(libs)

    n_unit = len(libs)
    i_idx, j_idx = _all_pairs_indices(n_unit)

    M_list = [len(lib["Q_sims"]) for lib in libs]
    same_M = (len(set(M_list)) == 1)

    cache = {
        "libs": libs,
        "obs_df": obs_df,
        "n_unit": n_unit,
        "i_idx": i_idx.astype(np.int32),
        "j_idx": j_idx.astype(np.int32),
        "same_M": same_M,
    }

    if same_M:
        M = int(M_list[0])

        Q_sims_mat = np.stack(
            [np.asarray(lib["Q_sims"], dtype=np.float64) for lib in libs],
            axis=0
        )
        P_sims_mat = np.stack(
            [np.asarray(lib["P_sims"], dtype=np.float64) for lib in libs],
            axis=0
        )

        cache["M"] = M
        cache["Q_sims_mat"] = Q_sims_mat
        cache["P_sims_mat"] = P_sims_mat

    else:
        cache["M_list"] = np.asarray(M_list, dtype=np.int32)
        cache["Q_sims_list"] = [
            np.asarray(lib["Q_sims"], dtype=np.float64) for lib in libs
        ]
        cache["P_sims_list"] = [
            np.asarray(lib["P_sims"], dtype=np.float64) for lib in libs
        ]

    return cache


def make_unit_rngs(n_unit, seed):
    """
    Create one independent RNG stream per unit.

    Using independent RNGs makes the sampled global realizations
    invariant to chunk_size, provided that the total N_global is fixed.
    """
    ss = np.random.SeedSequence(seed)
    child_seqs = ss.spawn(n_unit)
    return [np.random.default_rng(s) for s in child_seqs]


def sample_fast_block_chunk_invariant(cache, unit_rngs, n_blk):
    """
    Sample one block in a chunk-size-invariant way.

    Returns
    -------
    Q_blk : ndarray, shape (n_unit, n_blk)
    P_blk : ndarray, shape (n_unit, n_blk)
    """
    n_unit = cache["n_unit"]

    if cache["same_M"]:
        M = cache["M"]

        idx_blk = np.empty((n_unit, n_blk), dtype=np.int32)
        for d in range(n_unit):
            idx_blk[d, :] = unit_rngs[d].integers(0, M, size=n_blk, dtype=np.int32)

        Q_blk = np.take_along_axis(cache["Q_sims_mat"], idx_blk, axis=1)
        P_blk = np.take_along_axis(cache["P_sims_mat"], idx_blk, axis=1)

    else:
        Q_blk = np.empty((n_unit, n_blk), dtype=np.float64)
        P_blk = np.empty((n_unit, n_blk), dtype=np.float64)

        for d in range(n_unit):
            M = int(cache["M_list"][d])
            idx = unit_rngs[d].integers(0, M, size=n_blk, dtype=np.int32)

            Q_blk[d, :] = cache["Q_sims_list"][d][idx]
            P_blk[d, :] = cache["P_sims_list"][d][idx]

    return Q_blk, P_blk


# ============================================================
# G) Numba helpers
# ============================================================

@njit(fastmath=True)
def _upper_bound_float64(arr, x):
    """
    Return the first index k such that arr[k] > x.
    Equivalently, number of elements <= x.
    arr must be sorted ascending.
    """
    lo = 0
    hi = arr.shape[0]

    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= x:
            lo = mid + 1
        else:
            hi = mid

    return lo


@njit(fastmath=True)
def _lower_bound_float64(arr, x):
    """
    Return the first index k such that arr[k] >= x.
    arr must be sorted ascending.
    """
    lo = 0
    hi = arr.shape[0]

    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    return lo


# ============================================================
# H) Numba kernel 1: Tmax for each global realization
# ============================================================

@njit(parallel=True, fastmath=True)
def _tmax_best_block_kernel(
    Q_blk,
    P_blk,
    Q_ref_strong,
    Q_ref_weak,
    dP_ref,
    dP_floor
):
    """
    Compute T_max for each realization in a block.

    Definitions
    -----------
    R_ab = ordered pair-significance score
    S_ab = period-closeness score
    T_ab = min(R_ab, S_ab)

    T_max = max_{a<b} T_ab
    """
    n_unit, n_blk = Q_blk.shape

    inv_ref_strong = 1.0 / Q_ref_strong
    inv_ref_weak = 1.0 / Q_ref_weak

    tmax_best = np.zeros(n_blk, dtype=np.float32)

    for b in prange(n_blk):

        best = 0.0

        for i in range(n_unit - 1):

            Qi = Q_blk[i, b]
            Pi = P_blk[i, b]

            for j in range(i + 1, n_unit):

                Qj = Q_blk[j, b]
                Pj = P_blk[j, b]

                # R_ab
                if Qi >= Qj:
                    Qstrong = Qi
                    Qweak = Qj
                else:
                    Qstrong = Qj
                    Qweak = Qi

                r_strong = Qstrong * inv_ref_strong
                r_weak = Qweak * inv_ref_weak

                if r_strong <= r_weak:
                    R_ab = r_strong
                else:
                    R_ab = r_weak

                # S_ab
                dP = Pi - Pj
                if dP < 0.0:
                    dP = -dP

                denom = dP
                if denom < dP_floor:
                    denom = dP_floor

                S_ab = dP_ref / denom

                # T_ab = min(R_ab, S_ab)
                if R_ab <= S_ab:
                    T_ab = R_ab
                else:
                    T_ab = S_ab

                if T_ab > best:
                    best = T_ab

        tmax_best[b] = best

    return tmax_best


# ============================================================
# I) Numba kernel 2: existence grid in the Q-S plane
# ============================================================

@njit(parallel=True, fastmath=True)
def _q_s_grid_exceed_counts_block_kernel(
    Q_blk,
    P_blk,
    R_thr_grid,
    S_thr_grid,
    Q_ref_strong,
    Q_ref_weak,
    dP_ref,
    dP_floor
):
    """
    Compute existence counts on the (R_thr, S_thr) grid for one block.

    For each global realization b and each S threshold,
    we compute the maximum ordered pair-significance score R_ab
    among all pairs with S_ab >= S_thr_grid[j].
    Then for each R threshold, the existence event is true if

        max_R(S >= S_thr) >= R_thr.

    Returns
    -------
    counts : ndarray, shape (N_R, N_S)
        Number of realizations in this block satisfying the existence event.
    """
    n_unit, n_blk = Q_blk.shape
    N_R = R_thr_grid.shape[0]
    N_S = S_thr_grid.shape[0]

    inv_ref_strong = 1.0 / Q_ref_strong
    inv_ref_weak = 1.0 / Q_ref_weak

    n_threads = get_num_threads()
    tls_counts = np.zeros((n_threads, N_R, N_S), dtype=np.int64)

    for b in prange(n_blk):

        tid = get_thread_id()

        # exact_maxR[k] stores max ordered-pair significance score R_ab
        # for pairs whose largest accepted S-bin is k.
        exact_maxR = np.zeros(N_S, dtype=np.float64)
        exact_any = np.zeros(N_S, dtype=np.uint8)

        for i in range(n_unit - 1):

            Qi = Q_blk[i, b]
            Pi = P_blk[i, b]

            for j in range(i + 1, n_unit):

                Qj = Q_blk[j, b]
                Pj = P_blk[j, b]

                # R_ab
                if Qi >= Qj:
                    Qstrong = Qi
                    Qweak = Qj
                else:
                    Qstrong = Qj
                    Qweak = Qi

                r_strong = Qstrong * inv_ref_strong
                r_weak = Qweak * inv_ref_weak

                if r_strong <= r_weak:
                    R_ab = r_strong
                else:
                    R_ab = r_weak

                # S_ab
                dP = Pi - Pj
                if dP < 0.0:
                    dP = -dP

                denom = dP
                if denom < dP_floor:
                    denom = dP_floor

                S_ab = dP_ref / denom

                # Pair contributes to all S thresholds <= S_ab.
                n_accept = _upper_bound_float64(S_thr_grid, S_ab)

                if n_accept <= 0:
                    continue

                k_last = n_accept - 1

                if R_ab > exact_maxR[k_last]:
                    exact_maxR[k_last] = R_ab

                exact_any[k_last] = 1

        # Reverse cumulative over S-axis:
        # cum_maxR[js] = max R_ab among all pairs with S_ab >= S_thr_grid[js].
        cum_max = 0.0
        cum_any = 0

        for js in range(N_S - 1, -1, -1):

            if exact_any[js] != 0:
                cum_any = 1

            if exact_maxR[js] > cum_max:
                cum_max = exact_maxR[js]

            if cum_any == 0:
                continue

            # all R thresholds <= cum_max pass
            n_r_pass = _upper_bound_float64(R_thr_grid, cum_max)

            for ir in range(n_r_pass):
                tls_counts[tid, ir, js] += 1

    # Reduce thread-local counts
    counts = np.zeros((N_R, N_S), dtype=np.int64)

    for t in range(n_threads):
        for ir in range(N_R):
            for js in range(N_S):
                counts[ir, js] += tls_counts[t, ir, js]

    return counts


# ============================================================
# J) Single working-point global analysis using T_max
# ============================================================

def global_pair_single_working_point_fast(
    libs,
    target_pair_unitidx,
    N_global=100000,
    chunk_size=2000,
    seed=2027,
    store_null_distributions=True,
):
    """
    Single working-point global analysis using T_max.

    Definitions
    -----------
    For each analysis unit i,

        Q_i = -log10(p_i)

    For each pair (i,j),

        R_ij = min(max(Q_i,Q_j)/Q_obs,strong,
                   min(Q_i,Q_j)/Q_obs,weak)

        S_ij = dP_obs / max(|P_i - P_j|, dP_floor)

        T_ij = min(R_ij, S_ij)

    Global statistic:

        T_max = max_{i<j} T_ij

    Global false-alarm probability:

        p_T = N(T_max_sim >= T_max_obs) / N_global

    """
    dP_floor = 1e-5

    cache = prepare_global_fast_cache(libs)

    obs_df = cache["obs_df"]
    n_unit = cache["n_unit"]
    i_idx = cache["i_idx"]
    j_idx = cache["j_idx"]

    row_of_unit = {
        int(obs_df.loc[k, "unit_idx"]): k
        for k in range(len(obs_df))
    }

    u1, u2 = target_pair_unitidx

    if u1 not in row_of_unit or u2 not in row_of_unit:
        raise ValueError("target_pair_unitidx not found in libraries.")

    row1 = row_of_unit[u1]
    row2 = row_of_unit[u2]

    # Observed arrays
    Q_obs_all = obs_df["Q_obs"].to_numpy(np.float64)
    P_obs_all = obs_df["P_obs"].to_numpy(np.float64)

    # Reference values from target pair
    dP_obs = float(abs(P_obs_all[row1] - P_obs_all[row2]))

    if dP_floor > dP_obs:
        raise ValueError(
            f"dP_floor={dP_floor} is larger than dP_obs={dP_obs}. "
            "This would make the target pair S different from 1."
        )

    Q_ref_strong = float(max(Q_obs_all[row1], Q_obs_all[row2]))
    Q_ref_weak = float(min(Q_obs_all[row1], Q_obs_all[row2]))

    if Q_ref_weak <= 0:
        raise ValueError("Q_ref_weak must be positive.")

    if Q_ref_strong <= 0:
        raise ValueError("Q_ref_strong must be positive.")

    if Q_ref_strong < Q_ref_weak:
        raise ValueError("Q_ref_strong must be >= Q_ref_weak.")

    # Observed T_ij for all observed pairs
    Q1_obs = Q_obs_all[i_idx]
    Q2_obs = Q_obs_all[j_idx]

    P1_obs = P_obs_all[i_idx]
    P2_obs = P_obs_all[j_idx]

    R_obs_pairs = pair_R_from_Q(
        Q1_obs,
        Q2_obs,
        Q_ref_strong=Q_ref_strong,
        Q_ref_weak=Q_ref_weak
    )

    S_obs_pairs = pair_S_from_periods(
        P1_obs,
        P2_obs,
        dP_ref=dP_obs,
        dP_floor=dP_floor,
    )

    T_obs_pairs = np.minimum(R_obs_pairs, S_obs_pairs)

    T_max_obs = float(np.max(T_obs_pairs)) if T_obs_pairs.size > 0 else 0.0

    # Diagnostics: target-pair T
    target_mask = (
        ((i_idx == row1) & (j_idx == row2)) |
        ((i_idx == row2) & (j_idx == row1))
    )

    if np.any(target_mask):
        target_k = int(np.where(target_mask)[0][0])

        T_target_pair_obs = float(T_obs_pairs[target_k])
        R_target_pair_obs = float(R_obs_pairs[target_k])
        S_target_pair_obs = float(S_obs_pairs[target_k])
        dP_target_pair_obs = float(abs(P1_obs[target_k] - P2_obs[target_k]))
    else:
        T_target_pair_obs = np.nan
        R_target_pair_obs = np.nan
        S_target_pair_obs = np.nan
        dP_target_pair_obs = np.nan

    # Diagnostics: best observed T pair
    if T_obs_pairs.size > 0:
        idx_t_best_obs = int(np.argmax(T_obs_pairs))

        best_i_row = int(i_idx[idx_t_best_obs])
        best_j_row = int(j_idx[idx_t_best_obs])

        best_pair_unitidx = [
            int(obs_df.loc[best_i_row, "unit_idx"]),
            int(obs_df.loc[best_j_row, "unit_idx"])
        ]

        best_pair_mjd = [
            int(obs_df.loc[best_i_row, "mjd"]),
            int(obs_df.loc[best_j_row, "mjd"])
        ]

        best_pair_P = [
            float(P_obs_all[best_i_row]),
            float(P_obs_all[best_j_row])
        ]

        best_pair_dP = float(abs(P_obs_all[best_i_row] - P_obs_all[best_j_row]))
        best_pair_R = float(R_obs_pairs[idx_t_best_obs])
        best_pair_S = float(S_obs_pairs[idx_t_best_obs])
        best_pair_T = float(T_obs_pairs[idx_t_best_obs])
    else:
        idx_t_best_obs = -1
        best_pair_unitidx = [-1, -1]
        best_pair_mjd = [-1, -1]
        best_pair_P = [np.nan, np.nan]
        best_pair_dP = np.nan
        best_pair_R = np.nan
        best_pair_S = np.nan
        best_pair_T = np.nan

    # Global null assembly
    unit_rngs = make_unit_rngs(n_unit, seed)

    if store_null_distributions:
        T_null_distribution = np.empty(N_global, dtype=np.float32)
    else:
        T_null_distribution = None

    exceed_T = 0
    done = 0

    pbar = tqdm.tqdm(
        total=N_global,
        desc="Global assembly (T_max only, fast+numba)"
    )

    while done < N_global:
        n_blk = min(chunk_size, N_global - done)

        Q_blk, P_blk = sample_fast_block_chunk_invariant(
            cache,
            unit_rngs,
            n_blk
        )

        # Numba-accelerated T_max for each realization
        tmax_best_blk = _tmax_best_block_kernel(
            Q_blk,
            P_blk,
            Q_ref_strong,
            Q_ref_weak,
            dP_obs,
            dP_floor
        )

        if store_null_distributions:
            T_null_distribution[done:done+n_blk] = tmax_best_blk
            exceed_T += int(np.count_nonzero(tmax_best_blk >= T_max_obs))
        else:
            exceed_T += int(np.count_nonzero(tmax_best_blk >= T_max_obs))

        done += n_blk
        pbar.update(n_blk)

    pbar.close()

    # p-value (kept identical to your previous convention)
    p_T = (exceed_T) / (N_global)
    sigma_T = float(p_to_sigma(p_T))

    meta = {
        "config": {
            "target_pair_unitidx": [int(u1), int(u2)],
            "target_pair_mjd": [
                int(obs_df.loc[row1, "mjd"]),
                int(obs_df.loc[row2, "mjd"])
            ],
            "N_global": int(N_global),
            "chunk_size": int(chunk_size),
            "seed": int(seed),
            "N_units": int(n_unit),
            "statistic": "T_max",
            "q_metric": "neg_logp",
            "store_null_distributions": bool(store_null_distributions),
        },

        "working_point": {
            "dP_obs": float(dP_obs),
            "Q_ref_strong": float(Q_ref_strong),
            "Q_ref_weak": float(Q_ref_weak),
        },

        "observed_global_stats": {
            "T_max_obs": float(T_max_obs),

            "T_target_pair_obs": float(T_target_pair_obs),
            "R_target_pair_obs": float(R_target_pair_obs),
            "S_target_pair_obs": float(S_target_pair_obs),
            "dP_target_pair_obs": float(dP_target_pair_obs),

            "best_T_pair_index": int(idx_t_best_obs),
            "best_T_pair_unitidx": best_pair_unitidx,
            "best_T_pair_mjd": best_pair_mjd,
            "best_T_pair_P": best_pair_P,
            "best_T_pair_dP": float(best_pair_dP),
            "best_T_pair_R": float(best_pair_R),
            "best_T_pair_S": float(best_pair_S),
            "best_T_pair_T": float(best_pair_T),
        },

        "results": {
            "p_T": float(p_T),
            "sigma_T": float(sigma_T),
            "n_exceed_T": int(exceed_T),
        }
    }

    if store_null_distributions:
        meta["results"].update({
            "null_T_mean": float(np.mean(T_null_distribution)),
            "null_T_std": (
                float(np.std(T_null_distribution, ddof=1))
                if N_global > 1 else 0.0
            ),
            "null_T_min": float(np.min(T_null_distribution)),
            "null_T_max": float(np.max(T_null_distribution)),
        })

    arr = {
        "obs_unit_idx": obs_df["unit_idx"].to_numpy(np.int32),
        "obs_mjd": obs_df["mjd"].to_numpy(np.int32),
        "obs_chi": obs_df["chi_obs"].to_numpy(np.float64),
        "obs_P": obs_df["P_obs"].to_numpy(np.float64),
        "obs_meanT": obs_df["meanT_obs"].to_numpy(np.float64),
        "obs_p": obs_df["p_obs"].to_numpy(np.float64),
        "obs_Q": obs_df["Q_obs"].to_numpy(np.float64),

        "obs_pair_i_row": i_idx.astype(np.int32),
        "obs_pair_j_row": j_idx.astype(np.int32),
        "obs_pair_R": R_obs_pairs.astype(np.float32),
        "obs_pair_S": S_obs_pairs.astype(np.float32),
        "obs_pair_T": T_obs_pairs.astype(np.float32),
    }

    if store_null_distributions:
        arr["T_null_distribution"] = T_null_distribution.astype(np.float32)

    return meta, arr, obs_df


# ============================================================
# K) Save / load single-result and figures
# ============================================================

def build_filename_from_global_single_meta(meta, prefix="GLOBALPAIR"):
    cfg = meta["config"]
    N_global = cfg["N_global"]
    seed = cfg["seed"]

    filename = f"{prefix}_N{N_global}_seed{seed}.npz"
    return filename



def save_global_single_result(
    outdir,
    filename,
    meta,
    arr,
    obs_df,
):
    os.makedirs(outdir, exist_ok=True)
    npz_path = os.path.join(outdir, filename)

    meta_json_str = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(npz_path, **arr, meta_json=meta_json_str)

    json_path = npz_path.replace(".npz", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    csv_path = npz_path.replace(".npz", "_obs_catalog.csv")
    obs_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


    print("Saved:")
    print("  NPZ :", npz_path)
    print("  JSON:", json_path)
    print("  CSV :", csv_path)

    return npz_path


def load_global_single_result(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    arr = {
        "obs_unit_idx": z["obs_unit_idx"].astype(np.int32),
        "obs_mjd": z["obs_mjd"].astype(np.int32),
        "obs_chi": z["obs_chi"].astype(np.float64),
        "obs_P": z["obs_P"].astype(np.float64),
        "obs_meanT": z["obs_meanT"].astype(np.float64),
        "obs_p": z["obs_p"].astype(np.float64),
        "obs_Q": z["obs_Q"].astype(np.float64),
        "obs_pair_i_row": z["obs_pair_i_row"].astype(np.int32),
        "obs_pair_j_row": z["obs_pair_j_row"].astype(np.int32),
        "obs_pair_R": z["obs_pair_R"].astype(np.float32),
        "obs_pair_S": z["obs_pair_S"].astype(np.float32),
        "obs_pair_T": z["obs_pair_T"].astype(np.float32),
    }
    if "T_null_distribution" in z.files:
        arr["T_null_distribution"] = z["T_null_distribution"].astype(np.float32)
    csv_path = npz_path.replace(".npz", "_obs_catalog.csv")
    obs_df = pd.read_csv(csv_path)
    return meta, arr, obs_df


# ============================================================
# L) 2D grid scan + save/load in the R-S plane
# ============================================================

def global_pair_grid_scan_2d_RS_fast(
    libs,
    R_thr_grid,
    S_thr_grid,
    target_pair_unitidx=None,
    N_global=20000,
    chunk_size=1000,
    seed=2028,
    dP_floor=1e-5,
):
    """
    Fast 2D R-S grid scan.

    Axes
    ----
    y-axis:
        R_thr

    x-axis:
        S_thr

    Pair quantities
    ---------------
    Q_i = -log10(p_i)

    R_ab =
        min(max(Q_a,Q_b)/Q_obs,strong,
            min(Q_a,Q_b)/Q_obs,weak)

    S_ab =
        dP_obs / max(|P_a - P_b|, dP_floor)

    Returned probability
    --------------------
    p_exist_grid(ir, js) =
        P_null[
            exists at least one pair (a,b) such that
            R_ab >= R_thr_grid[ir]
            and
            S_ab >= S_thr_grid[js]
        ]

    Relation to T
    -------------
    T_ab = min(R_ab, S_ab)

    Therefore, on the diagonal R_thr = S_thr = t,

        p_exist_grid(t, t) = P_null(T_max >= t).

    In particular,

        p_exist_grid(1, 1) = P_null(T_max >= 1).
    """
    cache = prepare_global_fast_cache(libs)

    obs_df = cache["obs_df"]
    n_unit = cache["n_unit"]

    R_thr_grid = np.unique(np.asarray(R_thr_grid, dtype=np.float64))
    S_thr_grid = np.unique(np.asarray(S_thr_grid, dtype=np.float64))

    if np.any(R_thr_grid < 0):
        raise ValueError("R_thr_grid must be non-negative.")

    if np.any(S_thr_grid < 0):
        raise ValueError("S_thr_grid must be non-negative.")

    if target_pair_unitidx is None:
        raise ValueError("target_pair_unitidx must be provided.")

    row_of_unit = {
        int(obs_df.loc[k, "unit_idx"]): k
        for k in range(len(obs_df))
    }

    u1, u2 = target_pair_unitidx

    if u1 not in row_of_unit or u2 not in row_of_unit:
        raise ValueError("target_pair_unitidx not found in libraries.")

    row1 = row_of_unit[u1]
    row2 = row_of_unit[u2]

    # Observed reference values
    P_obs = obs_df["P_obs"].to_numpy(np.float64)
    Q_obs = obs_df["Q_obs"].to_numpy(np.float64)
    p_obs = obs_df["p_obs"].to_numpy(np.float64)
    chi_obs = obs_df["chi_obs"].to_numpy(np.float64)

    dP_ref = float(abs(P_obs[row1] - P_obs[row2]))

    if dP_floor > dP_ref:
        raise ValueError(
            f"dP_floor={dP_floor} is larger than dP_ref={dP_ref}. "
            "For the target pair to have S=1, dP_floor must be <= dP_ref."
        )

    Q_ref_strong = float(max(Q_obs[row1], Q_obs[row2]))
    Q_ref_weak = float(min(Q_obs[row1], Q_obs[row2]))

    if Q_ref_weak <= 0 or Q_ref_strong <= 0:
        raise ValueError("Reference Q scores must be positive.")

    if Q_ref_strong < Q_ref_weak:
        raise ValueError("Q_ref_strong must be >= Q_ref_weak.")

    N_R = len(R_thr_grid)
    N_S = len(S_thr_grid)

    S_max = dP_ref / dP_floor

    marker = {
        "target_pair_unitidx": [int(u1), int(u2)],
        "target_pair_mjd": [
            int(obs_df.loc[row1, "mjd"]),
            int(obs_df.loc[row2, "mjd"])
        ],
        "R_ref": 1.0,
        "S_ref": 1.0,
        "dP_ref": float(dP_ref),
        "dP_floor": float(dP_floor),
        "S_max": float(S_max),
        "q_metric": "neg_logp",
        "p_ref_weak": float(max(p_obs[row1], p_obs[row2])),
        "p_ref_strong": float(min(p_obs[row1], p_obs[row2])),
        "chi_ref_weak": float(min(chi_obs[row1], chi_obs[row2])),
        "chi_ref_strong": float(max(chi_obs[row1], chi_obs[row2])),
        "Q_ref_weak": float(Q_ref_weak),
        "Q_ref_strong": float(Q_ref_strong),
    }

    # Global null assembly
    exceed_exist_grid = np.zeros((N_R, N_S), dtype=np.int64)

    unit_rngs = make_unit_rngs(n_unit, seed)
    done = 0

    pbar = tqdm.tqdm(
        total=N_global,
        desc="Global 2D R-S grid scan fast"
    )

    while done < N_global:
        n_blk = min(chunk_size, N_global - done)

        Q_blk, P_blk = sample_fast_block_chunk_invariant(
            cache,
            unit_rngs,
            n_blk
        )

        counts_blk = _q_s_grid_exceed_counts_block_kernel(
            Q_blk=Q_blk,
            P_blk=P_blk,
            R_thr_grid=R_thr_grid,
            S_thr_grid=S_thr_grid,
            Q_ref_strong=Q_ref_strong,
            Q_ref_weak=Q_ref_weak,
            dP_ref=dP_ref,
            dP_floor=dP_floor
        )

        exceed_exist_grid += counts_blk

        done += n_blk
        pbar.update(n_blk)

    pbar.close()

    p_exist_grid = (exceed_exist_grid) / (N_global)

    meta = {
        "config": {
            "grid_type": "R_S_fast_exist_only",
            "q_metric": "neg_logp",
            "period_closeness": "continuous_S",
            "N_global": int(N_global),
            "chunk_size": int(chunk_size),
            "seed": int(seed),
            "N_units": int(n_unit),
            "N_R": int(N_R),
            "N_S": int(N_S),
        },
        "marker": marker,
    }

    arr = {
        "R_thr_grid": R_thr_grid.astype(np.float64),
        "S_thr_grid": S_thr_grid.astype(np.float64),
        "exceed_exist_grid": exceed_exist_grid.astype(np.int64),
        "p_exist_grid": p_exist_grid.astype(np.float64),

        "obs_unit_idx": obs_df["unit_idx"].to_numpy(np.int32),
        "obs_mjd": obs_df["mjd"].to_numpy(np.int32),
        "obs_chi": obs_df["chi_obs"].to_numpy(np.float64),
        "obs_P": obs_df["P_obs"].to_numpy(np.float64),
        "obs_meanT": obs_df["meanT_obs"].to_numpy(np.float64),
        "obs_p": obs_df["p_obs"].to_numpy(np.float64),
        "obs_Q": obs_df["Q_obs"].to_numpy(np.float64),
        "obs_z": obs_df["z_obs"].to_numpy(np.float64),
    }

    return meta, arr, obs_df


def build_filename_from_global_RSgrid2d_meta(meta, prefix="GLOBALRSGRID2D"):
    cfg = meta["config"]

    N_global = cfg["N_global"]
    seed = cfg["seed"]
    N_R = cfg["N_R"]
    N_S = cfg["N_S"]

    filename = (
        f"{prefix}"
        f"_N{N_global}_seed{seed}_NR{N_R}_NS{N_S}.npz"
    )

    return filename


def save_global_RSgrid2d_result(
    outdir,
    filename,
    meta,
    arr,
    obs_df,
):
    os.makedirs(outdir, exist_ok=True)
    npz_path = os.path.join(outdir, filename)

    meta_json_str = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(npz_path, **arr, meta_json=meta_json_str)

    json_path = npz_path.replace(".npz", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    csv_path = npz_path.replace(".npz", "_obs_catalog.csv")
    obs_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


    print("Saved:")
    print("  NPZ :", npz_path)
    print("  JSON:", json_path)
    print("  CSV :", csv_path)

    return npz_path


def load_global_RSgrid2d_result(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))

    arr = {
        "R_thr_grid": z["R_thr_grid"].astype(np.float64),
        "S_thr_grid": z["S_thr_grid"].astype(np.float64),
        "exceed_exist_grid": z["exceed_exist_grid"].astype(np.int64),
        "p_exist_grid": z["p_exist_grid"].astype(np.float64),

        "obs_unit_idx": z["obs_unit_idx"].astype(np.int32),
        "obs_mjd": z["obs_mjd"].astype(np.int32),
        "obs_chi": z["obs_chi"].astype(np.float64),
        "obs_P": z["obs_P"].astype(np.float64),
        "obs_meanT": z["obs_meanT"].astype(np.float64),
        "obs_p": z["obs_p"].astype(np.float64),
        "obs_Q": z["obs_Q"].astype(np.float64),
        "obs_z": z["obs_z"].astype(np.float64),
    }

    csv_path = npz_path.replace(".npz", "_obs_catalog.csv")
    obs_df = pd.read_csv(csv_path)

    return meta, arr, obs_df

def plot_global_t_null_distribution(meta, arr, figsize=(8, 6), color="#6A5ACD", logy=False):
    tnull = np.asarray(arr["T_null_distribution"], dtype=np.float64)
    tobs = float(meta["observed_global_stats"]["T_max_obs"])
    p_t = float(meta["results"]["p_T"])
    sig_t = float(meta["results"]["sigma_T"])

    xmax = float(max(np.max(tnull), tobs, 1.2))
    bins = np.linspace(0.0, xmax * 1.05, 60)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        tnull,
        bins=bins,
        color=color,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label="Simulated Samples"
    )

    ax.axvline(
        tobs,
        color="crimson",
        linestyle="--",
        linewidth=2.2,
        label=fr"Observed $\mathcal{{T}}_{{\max}}={tobs:.1f}$"
    )

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$\mathcal{T}_{\max}$")
    ax.set_ylabel("Counts")

    txt = f"{sig_t:.1f}$\\sigma$"

    ax.text(
        1.05, 1e5, txt,
        fontsize=16,
    )

    ax.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    plt.grid()
    return fig


def plot_global_RS_grid_heatmap(
    meta,
    arr,
    log10_color=True,
    cmap="magma",
    figsize=(8, 6),
    mark_target=True,
    mark_t_obs=None,
    draw_diagonal=False,
    sigma_contours=(3, 4, 5,),
    contour_color="cyan",
    contour_lw=1.2,
    contour_ls="--",
):
    """
    Plot R-S global existence probability heatmap with optional sigma contours.
    """
    R_grid = np.asarray(arr["R_thr_grid"], dtype=np.float64)
    S_grid = np.asarray(arr["S_thr_grid"], dtype=np.float64)

    p_grid = np.asarray(arr["p_exist_grid"], dtype=np.float64)
    p_grid_clip = np.clip(p_grid, 1e-300, 1.0)

    if log10_color:
        plot_val = np.log10(p_grid_clip)
        cbar_label = r"$\log_{10}$(FAP)"
    else:
        plot_val = p_grid_clip
        cbar_label = r"FAP"

    def log_centers_to_edges(x):
        x = np.asarray(x, dtype=np.float64)

        if np.any(x <= 0):
            raise ValueError("x must be positive for logarithmic edges.")

        logx = np.log10(x)

        log_edges = np.empty(len(x) + 1, dtype=np.float64)
        log_edges[1:-1] = 0.5 * (logx[:-1] + logx[1:])
        log_edges[0] = logx[0] - 0.5 * (logx[1] - logx[0])
        log_edges[-1] = logx[-1] + 0.5 * (logx[-1] - logx[-2])

        return 10 ** log_edges

    def linear_centers_to_edges(y):
        y = np.asarray(y, dtype=np.float64)

        edges = np.empty(len(y) + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (y[:-1] + y[1:])
        edges[0] = y[0] - 0.5 * (y[1] - y[0])
        edges[-1] = y[-1] + 0.5 * (y[-1] - y[-2])

        return edges

    S_edges = log_centers_to_edges(S_grid)
    R_edges = linear_centers_to_edges(R_grid)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        S_edges,
        R_edges,
        plot_val,
        shading="auto",
        cmap=cmap,
        vmin=-8,
        vmax=-2
    )

    ax.set_xscale("log")

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label)

    ax.set_xlabel(r"$\mathcal{S}_{\rm th}$")
    ax.set_ylabel(r"$\mathcal{R}_{\rm th}$")
    # ax.set_title(r"Global existence probability in the $R$--$S$ plane")

    # Sigma contours
    if sigma_contours is not None and len(sigma_contours) > 0:
        sigma_contours = np.asarray(sigma_contours, dtype=float)
        p_levels = norm.sf(sigma_contours)

        if log10_color:
            contour_levels_raw = np.log10(p_levels)
        else:
            contour_levels_raw = p_levels

        sort_idx = np.argsort(contour_levels_raw)
        contour_levels_sorted = contour_levels_raw[sort_idx]
        sigma_sorted = sigma_contours[sort_idx]

        vmin = np.nanmin(plot_val)
        vmax = np.nanmax(plot_val)

        keep = (
            np.isfinite(contour_levels_sorted) &
            (contour_levels_sorted >= vmin) &
            (contour_levels_sorted <= vmax)
        )

        contour_levels_use = contour_levels_sorted[keep]
        sigma_use = sigma_sorted[keep]

        if contour_levels_use.size > 0:
            S_mesh, R_mesh = np.meshgrid(S_grid, R_grid)

            cs = ax.contour(
                S_mesh,
                R_mesh,
                plot_val,
                levels=contour_levels_use,
                colors=contour_color,
                linewidths=contour_lw,
                linestyles=contour_ls
            )

            fmt = {}
            for lev, sig in zip(contour_levels_use, sigma_use):
                fmt[lev] = rf"{sig:g}$\sigma$"

            manual_positions = [
                (0.08, 0.45),
                (0.2, 0.58),
                (0.8, 0.88),
            ]

            ax.clabel(
                cs,
                cs.levels,
                inline=True,
                fontsize=12,
                fmt=fmt,
                manual=manual_positions
            )

    # Mark observed target pair level (R, S) = (1, 1)
    if mark_target:
        ax.scatter(
            [1.0],
            [1.0],
            marker="*",
            s=220,
            c="cyan",
            edgecolors="black",
            linewidths=1.0,
            label=r"Observed MJD 59310-59347 pair",
            zorder=5
        )

        ax.axhline(
            1.0,
            color="white",
            linestyle=":",
            linewidth=1.1,
            alpha=0.7
        )

        ax.axvline(
            1.0,
            color="white",
            linestyle=":",
            linewidth=1.1,
            alpha=0.7
        )
    ax.set_xlim(np.min(S_grid), np.max(S_grid))
    ax.set_ylim(np.min(R_grid), np.max(R_grid))
    # Draw diagonal R = S if desired.
    # Only meaningful where S overlaps the plotted R range.
    if draw_diagonal:
        rmin = max(np.min(R_grid), np.min(S_grid))
        rmax = min(np.max(R_grid), np.max(S_grid))
        if rmax > rmin:
            rr = np.linspace(rmin, rmax, 200)
            ax.plot(
                rr, rr,
                color="white",
                lw=1.2,
                ls="-",
                alpha=0.8,
                label=r"$\mathcal{R}_{\rm th}=\mathcal{S}_{\rm th}$"
            )

    ax.grid()
    ax.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    return fig

def plot_q_null_distribution(libs, figsize=(8, 6)):

    # ============================================================
    # target units
    # ============================================================
    unit_idx_1 = 3    # MJD 59310
    unit_idx_2 = 35   # MJD 59347
    
    # fetch corresponding libs
    lib1 = [x for x in libs if x["unit_idx"] == unit_idx_1][0]
    lib2 = [x for x in libs if x["unit_idx"] == unit_idx_2][0]
    
    # ============================================================
    # merge all null single-epoch significance scores Q = -log10(p)
    # ============================================================
    
    all_Q_sims = np.concatenate([np.asarray(x["Q_sims"], dtype=np.float64) for x in libs])
    
    # observed Q values
    Q_obs_1 = float(lib1["Q_obs"])
    Q_obs_2 = float(lib2["Q_obs"])
    
    # pooled exceedance counts
    n_total = len(all_Q_sims)
    n_exc_1 = np.sum(all_Q_sims >= Q_obs_1)
    n_exc_2 = np.sum(all_Q_sims >= Q_obs_2)
    
    # pooled exceedance fractions
    p_mix_1 = (n_exc_1 ) / (n_total )
    p_mix_2 = (n_exc_2 ) / (n_total )
    
    # convert to one-sided Gaussian sigma
    z_mix_1 = norm.isf(np.clip(p_mix_1, 1e-300, 1.0))
    z_mix_2 = norm.isf(np.clip(p_mix_2, 1e-300, 1.0))
    
    # print(f"[Merged null in Q=-log10(p)] unit_idx={unit_idx_1}, Q_obs={Q_obs_1:.4f}, "
    #       f"exceedance count = {n_exc_1}/{n_total}, pooled tail = {p_mix_1:.6e}, "
    #       f"equiv Z = {z_mix_1:.3f} sigma")
    
    # print(f"[Merged null in Q=-log10(p)] unit_idx={unit_idx_2}, Q_obs={Q_obs_2:.4f}, "
    #       f"exceedance count = {n_exc_2}/{n_total}, pooled tail = {p_mix_2:.6e}, "
    #       f"equiv Z = {z_mix_2:.3f} sigma")
    
    # ============================================================
    # histogram of pooled Q
    # ============================================================
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # choose histogram range automatically but ensure observed points are visible
    xmin = 0.0
    xmax = max(np.percentile(all_Q_sims, 99.9), Q_obs_1, Q_obs_2) * 1.1
    bins = np.linspace(xmin, xmax, 61)
    
    ax.hist(
        all_Q_sims,
        bins=bins,
        alpha=0.65,
        color="steelblue",
        edgecolor="none",
        label="Simulated samples"
    )
    
    ax.axvline(
        Q_obs_1,
        color="r",
        ls="--",
        lw=2,
        label=(
            "Observed $\\mathcal{Q}$ on MJD 59310"
        )
    )
    
    ax.axvline(
        Q_obs_2,
        color="orange",
        ls="--",
        lw=2,
        label=(
            "Observed $\\mathcal{Q}$ on MJD 59347"
        )
    )
    
    ax.text(Q_obs_1*1.01, 2e4, f'{z_mix_1:.1f}$\\sigma$',
             fontsize=16)
    ax.text(Q_obs_2*1.01, 2e4, f'{z_mix_2:.1f}$\\sigma$',
             fontsize=16)
    
    
    ax.set_yscale("log")
    ax.set_xlabel("$\\mathcal{Q}$")
    ax.set_xlim(-0.2,4.9)
    ax.set_ylabel("Counts")
    ax.grid()
    ax.legend(loc='upper right',fontsize=16)
    plt.tight_layout()
    return fig


def save_binned_histogram_data_from_global_t_null(outdir_global, fname_global, save_path=None):
    """
    Extract the histogram-binned data from the large result file, saving only:
    - bin_edges
    - bin_counts
    - T_max_obs
    - p_T
    - sigma_T
    """
    input_path = outdir_global + '\\' + fname_global
    meta_single, arr_single, _ = load_global_single_result(input_path)
    tnull = np.asarray(arr_single["T_null_distribution"], dtype=np.float64)
    tobs = float(meta_single["observed_global_stats"]["T_max_obs"])
    p_t = float(meta_single["results"]["p_T"])
    sig_t = float(meta_single["results"]["sigma_T"])
    xmax = float(max(np.max(tnull), tobs, 1.2))
    bins = np.linspace(0.0, xmax * 1.05, 60)
    counts, edges = np.histogram(tnull, bins=bins)
    if save_path is None:
        input_file = Path(input_path)
        save_path = input_file.with_name(input_file.stem + "_histbins.npz")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        bin_edges=edges,
        bin_counts=counts,
        T_max_obs=tobs,
        p_T=p_t,
        sigma_T=sig_t
    )
    print(f"Binned histogram data saved to: {save_path}")
    return save_path


def load_global_t_null_binned_histogram_data(histbin_path):
    data = np.load(histbin_path)

    hist_data = {
        "bin_edges": data["bin_edges"],
        "bin_counts": data["bin_counts"],
        "T_max_obs": float(data["T_max_obs"]),
        "p_T": float(data["p_T"]),
        "sigma_T": float(data["sigma_T"]),
    }
    return hist_data



def plot_global_t_null_distribution_from_binned_data(
    hist_data,
    figsize=(8, 6),
    color="#6A5ACD",
    logy=False
):
    edges = np.asarray(hist_data["bin_edges"], dtype=np.float64)
    counts = np.asarray(hist_data["bin_counts"], dtype=np.float64)
    tobs = float(hist_data["T_max_obs"])
    p_t = float(hist_data["p_T"])
    sig_t = float(hist_data["sigma_T"])

    lefts = edges[:-1]
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        lefts,
        counts,
        width=widths,
        align="edge",
        color=color,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label="Simulated Samples"
    )

    ax.axvline(
        tobs,
        color="crimson",
        linestyle="--",
        linewidth=2.2,
        label=fr"Observed $\mathcal{{T}}_{{\max}}={tobs:.1f}$"
    )

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$\mathcal{T}_{\max}$")
    ax.set_ylabel("Counts")

    txt = f"{sig_t:.1f}$\\sigma$"

    ax.text(
        1.05, 1e5, txt,
        fontsize=16,
    )

    ax.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    plt.grid()

    return fig

#%%
# ============================================================
# M) Example main
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # 0) File paths
    # --------------------------------------------------------
    fast1_path = r"Data\20201124A\Burst_Table\FAST#1.csv"
    fast2_path = r"Data\20201124A\Burst_Table\FAST#2.csv"
    ugmrt_path = r"Data\20201124A\Burst_Table\uGMRT.csv"
    eff_path   = r"Data\20201124A\Burst_Table\Effelsberg.csv"

    fast1_win_txt = r"Data\20201124A\windows1\FRB20201124A_fast1_obswindow_TCB_LTT.txt"
    fast2_win_txt = r"Data\20201124A\windows1\FRB20201124A_fast2_obswindow_TDB.txt"
    ugmrt_win_txt = r"Data\20201124A\windows1\FRB20201124A_ugmrt_obswindow_TDB.txt"
    eff_win_txt   = r"Data\20201124A\windows1\FRB20201124A_effelsberg_obswindow_TDB.txt"

    # --------------------------------------------------------
    # 1) Directly load all window files
    # --------------------------------------------------------
    w_fast1 = load_windows_txt(fast1_win_txt)
    w_fast2 = load_windows_txt(fast2_win_txt)
    w_ugmrt = load_windows_txt(ugmrt_win_txt)
    w_eff   = load_windows_txt(eff_win_txt)

    # --------------------------------------------------------
    # 2) Read all burst tables
    # --------------------------------------------------------
    b_fast1 = Xreaddata(fast1_path, [0, 2, 6, 4], ['t', 's', 'w', 'f'], startrow=0)
    b_fast2 = Xreaddata(fast2_path, [2, 5, 4, 7], ['t', 's', 'w', 'f'], startrow=2)
    b_ugmrt = Xreaddata(ugmrt_path, [1, 4, 2, 5], ['t', 's', 'w', 'f'], startrow=0)
    b_eff   = Xreaddata(eff_path,   [1, 5, 3, 7], ['t', 's', 'w', 'f'], startrow=0)

    # --------------------------------------------------------
    # 3) Split each dataset into unit lists
    # --------------------------------------------------------
    b_ss_fast1 = burstverify_anyday(b_fast1, nummin=1, dnum=1, sep=0)
    b_ss_fast2 = burstverify_anyday(b_fast2, nummin=1, dnum=1, sep=0)
    b_ss_ugmrt = burstverify_anyday(b_ugmrt, nummin=1, dnum=1, sep=0)
    b_ss_eff   = burstverify_anyday(b_eff,   nummin=1, dnum=1, sep=0)

    print("N units from FAST1      :", len(b_ss_fast1[1]))
    print("N units from FAST2      :", len(b_ss_fast2[1]))
    print("N units from uGMRT      :", len(b_ss_ugmrt[1]))
    print("N units from Effelsberg :", len(b_ss_eff[1]))

    # --------------------------------------------------------
    # 4) Build global unit list + metadata + window mapping
    # --------------------------------------------------------
    unit_dfs = []
    windows_obj = {}
    unit_meta = []

    def append_units(b_ss, windows_arr, dataset_name):
        for local_idx, df_day in enumerate(b_ss[1]):
            global_idx = len(unit_dfs)

            unit_dfs.append(df_day.copy())
            windows_obj[global_idx] = windows_arr

            unit_meta.append({
                "dataset": dataset_name,
                "local_idx": local_idx,
            })

    append_units(b_ss_fast1, w_fast1, "FAST1")
    append_units(b_ss_fast2, w_fast2, "FAST2")
    append_units(b_ss_ugmrt, w_ugmrt, "uGMRT")
    append_units(b_ss_eff,   w_eff,   "Effelsberg")

    unit_meta_df = pd.DataFrame(unit_meta)
    unit_meta_df.insert(0, "unit_idx", np.arange(len(unit_meta_df), dtype=int))

    print("\nAll analysis units:")
    print(unit_meta_df)

    # --------------------------------------------------------
    # 5) Build unit specs
    # --------------------------------------------------------
    pstart, pend, step = 0.1, 100.0, 1e-5
    bins_num = 20
    cluster_threshold = 0.5

    specs, meta_spec, p_short, p_long = build_all_unit_specs(
        unit_dfs=unit_dfs,
        windows_obj=windows_obj,
        pstart=pstart,
        pend=pend,
        step=step,
        bins_num=bins_num,
        cluster_threshold=cluster_threshold,
        unit_meta_list=unit_meta
    )

    print("\nSpec meta:", meta_spec)
    print("Number of valid units:", len(specs))

#%%
    # --------------------------------------------------------
    # 6) Build and save unit libraries
    # --------------------------------------------------------
    outdir_daylib = r"MC_Samples_Main\\UnitLibraries_allunits"
    N_mc_day = 10   # if set to 100000, runtime would be ~ 2 day
    sim_chunk_size = max(1, N_mc_day // 20)

    paths = build_and_save_all_unit_libraries(
        specs=specs,
        p_short=p_short,
        p_long=p_long,
        outdir=outdir_daylib,
        bins_num=bins_num,
        cluster_threshold=cluster_threshold,
        N_mc=N_mc_day,
        sim_chunk_size=sim_chunk_size,
        base_seed=2026
    )
    print("Saved libraries:", len(paths))
    
#%%
    outdir_daylib = r"MC_Samples_Main\\UnitLibraries_allunits"
    # --------------------------------------------------------
    # 7) Load + calibrate libraries
    # --------------------------------------------------------
    libs = load_all_unit_libraries(outdir_daylib, baseseedx=2026, nlocal=100000)
    libs = calibrate_all_libraries(libs)
    obs_df = build_observed_catalog_from_libraries(libs)

    print("\nObserved catalog:")
    keep_cols = ["unit_idx", "dataset", "mjd", "P_obs", "chi_obs", "p_obs"]
    print(obs_df[keep_cols])

    # --------------------------------------------------------
    # 8) Determine target pair automatically
    #    Example: FAST1 MJD 59310 and FAST1 MJD 59347
    # --------------------------------------------------------
    cand1 = obs_df[(obs_df["dataset"] == "FAST1") & (obs_df["mjd"] == 59310)]
    cand2 = obs_df[(obs_df["dataset"] == "FAST1") & (obs_df["mjd"] == 59347)]

    if len(cand1) != 1 or len(cand2) != 1:
        raise RuntimeError(
            "Cannot uniquely identify the target FAST1 units for MJD 59310 and 59347. "
            "Please inspect obs_df manually."
        )

    target_pair = (int(cand1.iloc[0]["unit_idx"]), int(cand2.iloc[0]["unit_idx"]))
    print("\nTarget pair unit_idx:", target_pair)

#%%
    # --------------------------------------------------------
    # 9) Single working-point global analysis
    # --------------------------------------------------------
    t0 = time.time()

    meta_single, arr_single, _ = global_pair_single_working_point_fast(
        libs=libs,
        target_pair_unitidx=target_pair,
        N_global=int(1e7), # if set to 1e9, runtime would be ~ 20 min
        chunk_size=int(1e5),
        seed=2026,
        store_null_distributions=True,
    )

    t1 = time.time()

    print("\nSingle-point global result:")
    print(json.dumps(meta_single, indent=2, ensure_ascii=False))
    print("Runtime(single-point):", t1 - t0, "s")
    
    # --------------------------------------------------------
    # Save single-point results 
    # --------------------------------------------------------
    outdir_global = r"MC_Samples_Main\\GlobalSingle_allunits"
    fname_global = build_filename_from_global_single_meta(meta_single, 
                                                          prefix="GLOBALSINGLE_ALLUNITS")

    _ = save_global_single_result(
        outdir=outdir_global,
        filename=fname_global,
        meta=meta_single,
        arr=arr_single,
        obs_df=obs_df,
    )

#%%

    # --------------------------------------------------------
    # 10) 2D Grid scan in the R-S plane
    # --------------------------------------------------------
    dP_ref = meta_single["working_point"]["dP_obs"]

    R_thr_grid = np.linspace(0.4, 1.2, 41)
    S_thr_grid = np.logspace(-2, 2, 41)
    
    # Ensure observed point is exactly included
    if np.any(np.isclose(R_thr_grid, 1.0)):
        R_thr_grid[np.isclose(R_thr_grid, 1.0)] = 1.0
    else:
        R_thr_grid = np.r_[R_thr_grid, 1.0]
    
    if np.any(np.isclose(S_thr_grid, 1.0)):
        S_thr_grid[np.isclose(S_thr_grid, 1.0)] = 1.0
    else:
        S_thr_grid = np.r_[S_thr_grid, 1.0]
    
    R_thr_grid = np.unique(R_thr_grid)
    S_thr_grid = np.unique(S_thr_grid)
    
    R_thr_grid.sort()
    S_thr_grid.sort()

    t2 = time.time()

    meta_grid, arr_grid, _ = global_pair_grid_scan_2d_RS_fast(
        libs=libs,
        R_thr_grid=R_thr_grid,
        S_thr_grid=S_thr_grid,
        target_pair_unitidx=target_pair,
        N_global=int(1e7), # if set to 1e9, runtime would be ~ 20 min
        chunk_size=int(1e5),
        seed=2026
    )

    t3 = time.time()

    print("\nGrid Scan result:")
    print(json.dumps(meta_grid, indent=2, ensure_ascii=False))
    print("Runtime(gridscan):", t3 - t2, "s")
    
    # --------------------------------------------------------
    #  Save gridscan results 
    # --------------------------------------------------------
    outdir_grid2d = r"MC_Samples_Main\\GlobalGrid2D_allunits"
    fname_grid2d = build_filename_from_global_RSgrid2d_meta(meta_grid, 
                                                            prefix="GLOBALRSGRID2D_ALLUNITS")
    
    _ = save_global_RSgrid2d_result(
        outdir=outdir_grid2d,
        filename=fname_grid2d,
        meta=meta_grid,
        arr=arr_grid,
        obs_df=obs_df,
    )


#%%
    # --------------------------------------------------------
    # 11) Plot single working-point null distribution
    # --------------------------------------------------------

    meta_single, arr_single,_=load_global_single_result(outdir_global+'\\'+fname_global)
    
    fig_t = plot_global_t_null_distribution(meta_single, arr_single, logy=1)
    output_file = Path(outdir_global+'\\'+fname_global)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.show()

    del fig_t
    gc.collect()
#%%
    # --------------------------------------------------------
    # 11) Plot single working-point null distribution (use hist_data)
    # --------------------------------------------------------
    histbin_file = save_binned_histogram_data_from_global_t_null(outdir_global, fname_global)
    
    
    hist_data = load_global_t_null_binned_histogram_data(histbin_file)
    
    fig_t = plot_global_t_null_distribution_from_binned_data(hist_data, logy=1)
    
    output_file = Path(outdir_global + '\\' + fname_global)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.show()
#%%
    # --------------------------------------------------------
    # 12) Plot heatmap in the R-S plane
    # --------------------------------------------------------


    meta_grid, arr_grid, _ = load_global_RSgrid2d_result(outdir_grid2d+'\\'+fname_grid2d)

    fig_grid = plot_global_RS_grid_heatmap(
        meta_grid,
        arr_grid,
        log10_color=True,
        mark_target=True,
        draw_diagonal=0,
    )
    output_file = Path(outdir_grid2d+'\\'+fname_grid2d)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file.with_suffix('.pdf'))
    plt.show()

    del fig_grid
    gc.collect()
#%%
    fig_q = plot_q_null_distribution(libs, figsize=(8, 6))
    output_file = Path(outdir_daylib+'\\'+'Q_ALL_EPOCH')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file.with_suffix('.pdf'))