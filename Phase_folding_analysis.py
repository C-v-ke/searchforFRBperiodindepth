# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 03:10:24 2026

@author: Cvke
"""
# Import necessary libraries and configure matplotlib parameters
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pandas as pd 
import numpy as np
import seaborn as sns
import math
import time
import tqdm
import gc
from numba import njit, prange
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import NullLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
from brokenaxes import brokenaxes
from pathlib import Path

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
# Read burst table
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

b_fast2=Xreaddata('Data\\20201124A\\Burst_Table\\FAST#2.csv',
            [2,5,4,7],['t','s','w','f'],
            startrow=2) 
b_fast2[1]['s']=b_fast2[1]['s']/1000

b_ugmrt=Xreaddata('Data\\20201124A\\Burst_Table\\uGMRT.csv',
            [1,4,2,5],['t','s','w','f'],
            startrow=0)  

b_effelsberg=Xreaddata('Data\\20201124A\\Burst_Table\\Effelsberg.csv',
            [1,5,3,7],['t','s','w','f'],
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
        # bi_ss[1][i]['waitingtime']=np.concatenate(([np.inf],np.diff(b_s[i])))
        bi_ss[1][i]['waitingtime0'] = np.concatenate(([np.inf], np.diff(b_s[i])))
        bi_ss[1][i]['waitingtime1'] = np.concatenate((np.diff(b_s[i]), [np.inf]))
    return bi_ss



#%%


# %%
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# =========================================================
# 1. basic tools
# =========================================================
def wrap_phase(phi):
    """
    Map the phase to [0, 1)
    """
    phi = np.fmod(phi, 1.0)
    phi = np.where(phi < 0.0, phi + 1.0, phi)
    return phi


def mjd_label_from_day(day_dict):
    """
    根据 day_dict['t'] 生成 MJD 标签
    """
    t_mjd = np.asarray(day_dict['t'], dtype=float)
    if np.max(t_mjd) - np.min(t_mjd) < 1.0:
        return f"MJD {int(np.floor(np.min(t_mjd)))}"
    else:
        return (
            f"MJD {int(np.floor(np.min(t_mjd)))}"
            f"-{int(np.floor(np.max(t_mjd)))}"
        )


def choose_t_ref(t_s, ref_mode="burst_mean", t_ref_user=None):
    """
    Select a reference time within a certain day
    ----
    t_s : array-like
        burst TOA, in seconds
    ref_mode : str
        "burst_mean" : use the burst mean TOA
        "mid_range"  : use (min+max)/2
    t_ref_user : float or None
        if provided, use this value directly as the reference time
    """
    t_s = np.asarray(t_s, dtype=float)

    if t_ref_user is not None:
        return float(t_ref_user)

    if ref_mode == "burst_mean":
        return np.mean(t_s)
    elif ref_mode == "mid_range":
        return 0.5 * (np.min(t_s) + np.max(t_s))
    else:
        raise ValueError(f"Unknown ref_mode: {ref_mode}")


# =========================================================
# 2. 相位计算：线性 P(t) = P_ref + Pdot * t
# =========================================================
def calculate_phase_vectorized(dt, period_ref, pdot, eps=1e-18):
    """
    Vectorized version of phase calculation
    
    Let:
    P(t) = P_ref + pdot * t
    Then the accumulated number of cycles:
    N(dt) = ln(1 + pdot*dt/P_ref) / pdot
    Phase:
    phi = N mod 1
    
    Parameters
    dt : ndarray, shape (n_bursts,)
    period_ref : ndarray, shape (n_mc,)
    pdot : ndarray, shape (n_mc,)
    eps : float
    If |pdot| < eps, use the constant-period approximation
    
    
    Returns
    phi_mc : ndarray, shape (n_mc, n_bursts)
    """
    dt = np.asarray(dt, dtype=float)
    period_ref = np.asarray(period_ref, dtype=float)
    pdot = np.asarray(pdot, dtype=float)

    n_mc = len(period_ref)
    n_bursts = len(dt)

    phi = np.empty((n_mc, n_bursts), dtype=float)

    small = np.abs(pdot) < eps
    large = ~small

    if np.any(large):
        p_large = period_ref[large, None]
        pd_large = pdot[large, None]
        x = pd_large * dt[None, :] / p_large
        phi[large, :] = np.log1p(x) / pd_large

    if np.any(small):
        phi[small, :] = dt[None, :] / period_ref[small, None]

    return wrap_phase(phi)


def sample_split_normal(mu, sigma_minus, sigma_plus, size, rng=None):
    """
    split normal / two-piece Gaussian approximate sampling:
    use sigma_minus when z < 0
    use sigma_plus when z >= 0
    """
    if rng is None:
        rng = np.random.default_rng()

    z = rng.standard_normal(size)
    x = mu + np.where(z < 0.0, z * sigma_minus, z * sigma_plus)
    return x


# =========================================================
# 3. 从两天锚点周期外推到某一天，做 MC 相位传播
# =========================================================
def compute_phi_mc_for_day(
    day_dict,
    t0_global,
    t1_global,
    P0_mean,
    P0_err_minus,
    P0_err_plus,
    P1_mean,
    P1_err_minus,
    P1_err_plus,
    n_mc=500,
    ref_mode="burst_mean",
    t_ref_user=None,
    random_seed=None
):
    """
    Generate MC phase samples phi_mc for a given day
    """
    rng = np.random.default_rng(random_seed)

    t_s = np.asarray(day_dict["t_s"], dtype=float)
    n_bursts = len(t_s)

    t_ref = choose_t_ref(t_s, ref_mode=ref_mode, t_ref_user=t_ref_user)
    dt_day = t_s - t_ref

    dt_10 = t1_global - t0_global
    dt_ref0 = t_ref - t0_global

    P0_samp = sample_split_normal(
        P0_mean, P0_err_minus, P0_err_plus, n_mc, rng=rng
    )
    P1_samp = sample_split_normal(
        P1_mean, P1_err_minus, P1_err_plus, n_mc, rng=rng
    )

    pdot_samp = (P1_samp - P0_samp) / dt_10
    Pref_samp = P0_samp + pdot_samp * dt_ref0

    phi_mc = calculate_phase_vectorized(
        dt=dt_day,
        period_ref=Pref_samp,
        pdot=pdot_samp
    )

    return {
        "phi_mc": phi_mc,
        "t_s": t_s,
        "t_ref": t_ref,
        "dt_day": dt_day,
        "dt_ref0": dt_ref0,
        "P0_samp": P0_samp,
        "P1_samp": P1_samp,
        "pdot_samp": pdot_samp,
        "Pref_samp": Pref_samp,
        "n_bursts": n_bursts
    }


# =========================================================
# 4. posterior predictive histogram / KDE
# =========================================================
def _is_uniform_bins(bins, rtol=1e-12, atol=1e-15):
    bins = np.asarray(bins, dtype=float)
    db = np.diff(bins)
    return np.allclose(db, db[0], rtol=rtol, atol=atol)



def posterior_predictive_hist(phi_mc, bins):
    """
    For each MC realization, make a histogram separately, 
    then compute the mean and std
    """
    phi_mc = np.asarray(phi_mc, dtype=float)
    bins = np.asarray(bins, dtype=float)

    n_mc, n_bursts = phi_mc.shape
    n_bin = len(bins) - 1

    if n_bin <= 0:
        raise ValueError("bins must contain at least two edges.")

    if _is_uniform_bins(bins):
        binw = bins[1] - bins[0]
        idx = np.floor((phi_mc - bins[0]) / binw).astype(np.int64)
        idx = np.clip(idx, 0, n_bin - 1)

        counts = np.zeros((n_mc, n_bin), dtype=np.int32)
        rows = np.repeat(np.arange(n_mc, dtype=np.int64), n_bursts)
        np.add.at(counts, (rows, idx.ravel()), 1)

        hist_mc = counts.astype(float) / (n_bursts * binw)
    else:
        hist_mc = np.array([
            np.histogram(phi_mc[k, :], bins=bins, density=True)[0]
            for k in range(n_mc)
        ])

    centers = 0.5 * (bins[:-1] + bins[1:])
    h_mean = np.mean(hist_mc, axis=0)
    h_std = np.std(hist_mc, axis=0, ddof=1)
    h_median = np.median(hist_mc, axis=0)
    h_lo = np.quantile(hist_mc, 0.16, axis=0)
    h_hi = np.quantile(hist_mc, 0.84, axis=0)

    return centers, h_mean, h_std, h_median, h_lo, h_hi, hist_mc

def circular_kde_phase(phases, grid_phase, kappa=20.0):
    """
    用 von Mises kernel 做 circular KDE
    """
    phases = np.asarray(phases, dtype=float)
    grid_phase = np.asarray(grid_phase, dtype=float)

    theta_g = 2.0 * np.pi * grid_phase[None, :]
    theta_d = 2.0 * np.pi * phases[:, None]

    vm = stats.vonmises.pdf(theta_g, kappa, loc=theta_d)
    kde_theta = np.mean(vm, axis=0)
    kde_phase = kde_theta * 2.0 * np.pi
    return kde_phase



def posterior_predictive_kde(phi_mc, grid_phase, kappa=20.0):
    """
    For each MC realization, perform circular KDE, 
    then compute the mean and std
    """
    phi_mc = np.asarray(phi_mc, dtype=float)

    kde_mc = np.array([
        circular_kde_phase(phi_mc[k, :], grid_phase, kappa=kappa)
        for k in range(phi_mc.shape[0])
    ])

    kde_mean = np.mean(kde_mc, axis=0)
    kde_std = np.std(kde_mc, axis=0, ddof=1)
    kde_median = np.median(kde_mc, axis=0)
    kde_lo = np.quantile(kde_mc, 0.16, axis=0)
    kde_hi = np.quantile(kde_mc, 0.84, axis=0)

    return {
        "grid_phase": np.asarray(grid_phase, dtype=float),
        "kde_mean": kde_mean,
        "kde_std": kde_std,
        "kde_median": kde_median,
        "kde_lo": kde_lo,
        "kde_hi": kde_hi,
        "kde_mc": kde_mc
    }
# =========================================================
# 5. H-test
# =========================================================
@njit(fastmath=True)
def h_test_stat_numba_single(phi, mmax=20):
    """
    Compute the H-test for a single phase sample
    The input phi must already be within [0,1)
    """
    n = len(phi)
    two_pi = 2.0 * np.pi

    sum_c = np.zeros(mmax, dtype=np.float64)
    sum_s = np.zeros(mmax, dtype=np.float64)

    for i in range(n):
        theta = two_pi * phi[i]

        c1 = np.cos(theta)
        s1 = np.sin(theta)

        sum_c[0] += c1
        sum_s[0] += s1

        cmx = c1
        smx = s1
        for h in range(1, mmax):
            c_new = cmx * c1 - smx * s1
            s_new = smx * c1 + cmx * s1

            sum_c[h] += c_new
            sum_s[h] += s_new

            cmx = c_new
            smx = s_new

    cumulative_Z2 = 0.0
    max_H = -1e100
    m_best = 1

    for h in range(mmax):
        Zm2 = 2.0 * (sum_c[h] * sum_c[h] + sum_s[h] * sum_s[h]) / n
        cumulative_Z2 += Zm2
        current_H = cumulative_Z2 - 4.0 * (h + 1) + 4.0

        if current_H > max_H:
            max_H = current_H
            m_best = h + 1

    return max_H, m_best


@njit(parallel=True, fastmath=True)
def h_test_stat_numba_batch(phi_mat, mmax=20):
    """
    Batch compute the H-test for a phase matrix with shape=(n_sim, n)
    Each row is a set of phase samples
    Returns:
    H_array: shape=(n_sim,)
    """
    n_sim, n = phi_mat.shape
    two_pi = 2.0 * np.pi
    H_array = np.empty(n_sim, dtype=np.float64)

    for k in prange(n_sim):
        sum_c = np.zeros(mmax, dtype=np.float64)
        sum_s = np.zeros(mmax, dtype=np.float64)

        for i in range(n):
            theta = two_pi * phi_mat[k, i]

            c1 = np.cos(theta)
            s1 = np.sin(theta)

            sum_c[0] += c1
            sum_s[0] += s1

            cmx = c1
            smx = s1
            for h in range(1, mmax):
                c_new = cmx * c1 - smx * s1
                s_new = smx * c1 + cmx * s1

                sum_c[h] += c_new
                sum_s[h] += s_new

                cmx = c_new
                smx = s_new

        cumulative_Z2 = 0.0
        max_H = -1e100

        for h in range(mmax):
            Zm2 = 2.0 * (sum_c[h] * sum_c[h] + sum_s[h] * sum_s[h]) / n
            cumulative_Z2 += Zm2
            current_H = cumulative_Z2 - 4.0 * (h + 1) + 4.0

            if current_H > max_H:
                max_H = current_H

        H_array[k] = max_H

    return H_array


def h_test_stat(phi, mmax=20):
    """
    H-test stat
    """
    phi = wrap_phase(np.asarray(phi, dtype=np.float64))
    h_best, m_best = h_test_stat_numba_single(phi, mmax=mmax)
    return float(h_best), int(m_best)


def sample_null_stats_h(n, n_sim=20000, random_seed=None, mmax=20):
    """
    Simulate the null distribution of the H-test under H0: phi ~ Uniform(0,1) 
    """
    rng = np.random.default_rng(random_seed)
    phi_null = rng.random((n_sim, n)).astype(np.float64)
    h_best = h_test_stat_numba_batch(phi_null, mmax=mmax)
    return h_best



def _tail_pvalues_from_sorted_null(obs_stats, null_sorted):
    """
    Given a sorted null distribution, 
    quickly compute the tail p-values for a batch of obs_stats
    """
    obs_stats = np.asarray(obs_stats, dtype=float)
    idx = np.searchsorted(null_sorted, obs_stats, side="left")
    pvals = (len(null_sorted) - idx) / (len(null_sorted) + 1.0)
    return pvals


def test_uniformity_phi_mc(
    phi_mc,
    tests=("h",),
    n_null=20000,
    random_seed=None,
    mmax=20,
    bins_num=20
):
    """
    Perform a uniformity test on each MC realization of phi_mc, and summarize the results
    """
    phi_mc = np.asarray(phi_mc, dtype=float)
    n_mc, n = phi_mc.shape

    results = {}

    # -------------------------
    # H-test
    # -------------------------
    if "h" in tests:
        obs_h = np.array([
            h_test_stat(phi_mc[k], mmax=mmax)[0] for k in range(n_mc)
        ], dtype=float)

        null_h = sample_null_stats_h(
            n=n,
            n_sim=n_null,
            random_seed=random_seed,
            mmax=mmax
        )
        null_h_sorted = np.sort(null_h)
        p_h = _tail_pvalues_from_sorted_null(obs_h, null_h_sorted)

        results["h"] = {
            "obs_stats": obs_h,
            "null_stats": null_h,
            "pvalues": p_h,
            "summary": {
                "median_stat": float(np.median(obs_h)),
                "q16_stat": float(np.quantile(obs_h, 0.16)),
                "q84_stat": float(np.quantile(obs_h, 0.84)),
                "median_p": float(np.median(p_h)),
                "q16_p": float(np.quantile(p_h, 0.16)),
                "q84_p": float(np.quantile(p_h, 0.84)),
                "reject_frac_0.1": float(np.mean(p_h < 0.1))
            }
        }


    return results


# =========================================================
# 6. Plot figures
# =========================================================
def _duplicate_for_two_cycles(x, y):
    x2 = np.concatenate([x, x + 1.0])
    y2 = np.concatenate([y, y])
    return x2, y2



def format_uniformity_summary_text(mc_results=None):
    """
    Significance summary text in the figure title
    """
    if mc_results is None:
        return ""

    parts = []

    if "h" in mc_results:
        s = mc_results["h"]["summary"]
        parts.append(
            f"H-test median p={s['median_p']:.3g}, fraction(p<0.1)={s['reject_frac_0.1']:.2f}"
        )

    if len(parts) == 0:
        return ""

    return "Monte Carlo simulation: " + " | ".join(parts)

# =========================================================
# 7. Plot single-day phase distribution
# =========================================================

def plot_phase_uniformity(
    day_dict,
    phi_mc,
    ds="FAST",
    bins=np.linspace(0.0, 1.0, 13),
    draw_kde=True,
    kde_summary=None,
    mc_results=None,
    rc_dict=None,
    figsize=(8, 6)
):
    """
    Single-day phase distribution plotting:
    
    mean ± 1σ interval of the posterior predictive histogram
    
    mean ± 1σ interval of the circular KDE (optional)
    """
    if rc_dict is not None:
        plt.rcParams.update(rc_dict)

    centers, h_mean, h_std, _, _, _, _ = posterior_predictive_hist(phi_mc, bins=bins)
    h_lo_meanc = np.maximum(h_mean - h_std, 0.0)
    h_hi_meanc = h_mean + h_std
    ymax = np.max(h_hi_meanc) if len(h_hi_meanc) > 0 else 0.0

    use_kde = draw_kde and (kde_summary is not None)
    if use_kde:
        grid_phase = kde_summary["grid_phase"]
        kde_mean = kde_summary["kde_mean"]
        kde_std = kde_summary["kde_std"]
        kde_lo_meanc = np.maximum(kde_mean - kde_std, 0.0)
        kde_hi_meanc = kde_mean + kde_std
        ymax = max(ymax, np.max(kde_hi_meanc) if len(kde_hi_meanc) > 0 else 0.0)

    ymax = max(1.05, 1.25 * ymax)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    mjd_text = mjd_label_from_day(day_dict)
    n_bursts = len(day_dict["t_s"])

    ax.text(
        0.05, 0.95, f"{ds} {mjd_text}",
        transform=ax.transAxes, fontsize=16,
        verticalalignment='top', horizontalalignment='left'
    )
    ax.text(
        0.05, 0.9, f"{n_bursts} bursts",
        transform=ax.transAxes, fontsize=16,
        verticalalignment='top', horizontalalignment='left'
    )
    fig.suptitle(format_uniformity_summary_text(mc_results=mc_results), fontsize=12)

    binw = bins[1] - bins[0]

    centers2, h_mean2 = _duplicate_for_two_cycles(centers, h_mean)
    ax.bar(
        centers2,
        h_mean2,
        width=binw,
        color="royalblue",
        alpha=0.5,
        edgecolor="none",
        align="center",
    )

    edges2 = np.concatenate([bins, bins[1:] + 1.0])
    h_lo2 = np.concatenate([h_lo_meanc, h_lo_meanc])
    h_hi2 = np.concatenate([h_hi_meanc, h_hi_meanc])

    x_step = np.repeat(edges2, 2)[1:-1]
    y_lo_step = np.repeat(h_lo2, 2)
    y_hi_step = np.repeat(h_hi2, 2)

    ax.fill_between(
        x_step,
        y_lo_step,
        y_hi_step,
        color="orange",
        alpha=0.15,
        edgecolor="none",
        linewidth=0,
    )

    for i in range(len(h_lo2)):
        x0 = edges2[i]
        x1 = edges2[i + 1]
        y0 = h_lo2[i]
        y1 = h_hi2[i]

        ax.hlines(y0, x0, x1, color="darkorange", lw=1, alpha=0.3)
        ax.hlines(y1, x0, x1, color="darkorange", lw=1, alpha=0.3)
        ax.vlines(x0, y0, y1, color="darkorange", lw=1, alpha=0.3)
        ax.vlines(x1, y0, y1, color="darkorange", lw=1, alpha=0.3)

    if use_kde:
        grid2, kde_mean2 = _duplicate_for_two_cycles(grid_phase, kde_mean)
        _, kde_lo2 = _duplicate_for_two_cycles(grid_phase, kde_lo_meanc)
        _, kde_hi2 = _duplicate_for_two_cycles(grid_phase, kde_hi_meanc)

        ax.plot(
            grid2,
            kde_mean2,
            color="black",
            lw=1.8,
            alpha=0.75
        )

        ax.fill_between(
            grid2,
            kde_lo2,
            kde_hi2,
            color="gray",
            alpha=0.36,
            edgecolor="k",
        )

    ax.set_xlim(0.0, 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))

    legend_handles = [
        Patch(
            facecolor="royalblue", alpha=0.5, edgecolor="none",
            label="Histogram mean"
        ),
        Patch(
            facecolor="orange", alpha=0.15, edgecolor="darkorange",
            label="Histogram $\\pm1\\sigma$"
        ),
    ]

    if use_kde:
        legend_handles += [
            Line2D(
                [0], [0], color="black", lw=1.8, alpha=0.75,
                label="KDE mean"
            ),
            Patch(
                facecolor="gray", alpha=0.3, edgecolor="k",
                label="KDE $\\pm1\\sigma$"
            ),
        ]

    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        fontsize=15,
        frameon=1,
        ncol=2,
        columnspacing=0.5
    )

    return fig

# =========================================================
# 8. Analyze
# =========================================================
def analyze_one_day_uniformity(
    day_dict,
    ds,
    t0_global,
    t1_global,
    P0_mean,
    P0_err_minus,
    P0_err_plus,
    P1_mean,
    P1_err_minus,
    P1_err_plus,
    n_mc=300,
    ref_mode="burst_mean",
    t_ref_user=None,
    random_seed=None,
    figsize=(8, 6),
    bins=np.linspace(0.0, 1.0, 13),
    draw_kde=True,
    kde_grid=np.linspace(0.0, 1.0, 400, endpoint=False),
    kappa_kde=20.0,
    do_uniformity_tests=True,
    tests=("h",),
    n_null=20000,
    mmax_h=20,
    nbins=20,
    save_dir=None,
    file_tag="",
    rc_dict=None
):
    """
    One-stop completion:
    
    Period error -> phase MC propagation
    
    Optional: posterior predictive KDE
    
    Perform a uniformity test on phi_mc
    
    Plot a single-day phase distribution
    """
    mc_result = compute_phi_mc_for_day(
        day_dict=day_dict,
        t0_global=t0_global,
        t1_global=t1_global,
        P0_mean=P0_mean,
        P0_err_minus=P0_err_minus,
        P0_err_plus=P0_err_plus,
        P1_mean=P1_mean,
        P1_err_minus=P1_err_minus,
        P1_err_plus=P1_err_plus,
        n_mc=n_mc,
        ref_mode=ref_mode,
        t_ref_user=t_ref_user,
        random_seed=random_seed
    )

    phi_mc = mc_result["phi_mc"]

    if draw_kde:
        kde_summary = posterior_predictive_kde(
            phi_mc=phi_mc,
            grid_phase=kde_grid,
            kappa=kappa_kde
        )
    else:
        kde_summary = None

    mc_uniformity_results = None
    if do_uniformity_tests:
        mc_uniformity_results = test_uniformity_phi_mc(
            phi_mc=phi_mc,
            tests=tests,
            n_null=n_null,
            random_seed=random_seed,
            mmax=mmax_h,
            bins_num=nbins
        )

    fig = plot_phase_uniformity(
        day_dict=day_dict,
        phi_mc=phi_mc,
        ds=ds,
        bins=bins,
        draw_kde=draw_kde,
        kde_summary=kde_summary,
        mc_results=mc_uniformity_results,
        rc_dict=rc_dict,
        figsize=figsize
    )

    output_file = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ds=ds.replace(" ", "_")
        mjd_text = mjd_label_from_day(day_dict).replace(" ", "_")
        suffix = f"_{file_tag}" if file_tag else ""
        output_file = save_dir / f"{ds}_{mjd_text}{suffix}.svg"
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

    return {
        "mc_result": mc_result,
        "kde_summary": kde_summary,
        "uniformity_results": mc_uniformity_results,
        "mc_uniformity_results": mc_uniformity_results,
        "hist_null_env": None,
        "kde_null_env": None,
        "fig": fig,
        "output_file": output_file
    }


# =========================================================
# 9. Print summary of uniformity test
# =========================================================

def print_uniformity_summary(result_dict):
    """
    Print the test summary of analyze_one_day_uniformity
    """
    mc_results = result_dict.get("mc_uniformity_results", None)

    if mc_results is None:
        print("Uniformity tests skipped.")
        return

    print("=" * 72)
    print("Uniformity test summary: MC over ephemeris")
    print("-" * 72)

    if "h" in mc_results:
        s = mc_results["h"]["summary"]
        print("H-test:")
        print(f"  median stat     = {s['median_stat']:.4f}")
        print(f"  stat [16%,84%]  = [{s['q16_stat']:.4f}, {s['q84_stat']:.4f}]")
        print(f"  median p        = {s['median_p']:.4g}")
        print(f"  p [16%,84%]     = [{s['q16_p']:.4g}, {s['q84_p']:.4g}]")
        print(f"  frac(p < 0.1)   = {s['reject_frac_0.1']:.3f}")
        print("-" * 72)

    print("=" * 72)
    
# =========================================================
# 10. Example
# =========================================================

if __name__ == "__main__":
    b_ss_fast1 = burstverify_anyday(b_fast1, nummin=1, dnum=1, sep=0)
    b_ss_fast2 = burstverify_anyday(b_fast2, nummin=1, dnum=1, sep=0)
    b_ss_ugmrt = burstverify_anyday(b_ugmrt, nummin=1, dnum=1, sep=0)
    b_ss_eff = burstverify_anyday(b_effelsberg, nummin=1, dnum=1, sep=0)
    
    t0_global = np.mean(b_ss_fast1[1][3]['t_s'])   # 59310
    t1_global = np.mean(b_ss_fast1[1][35]['t_s'])  # 59347
    
    P0_mean = 1.706024
    P0_err_minus = 0.000013
    P0_err_plus = 0.000013

    P1_mean = 1.707968
    P1_err_minus = 0.000009
    P1_err_plus = 0.000009
    
    for i in tqdm.tqdm([7]):
        b_sss = b_ss_fast1[1][i]
    
    
        result = analyze_one_day_uniformity(
            day_dict=b_sss,
            ds="FAST #1",
            t0_global=t0_global,
            t1_global=t1_global,
            P0_mean=P0_mean,
            P0_err_minus=P0_err_minus,
            P0_err_plus=P0_err_plus,
            P1_mean=P1_mean,
            P1_err_minus=P1_err_minus,
            P1_err_plus=P1_err_plus,
            
    
            # phi MC
            n_mc=10000,
            ref_mode="burst_mean",
            t_ref_user=None,
            random_seed=None,
    
            # figure
            figsize=(8, 6),
            bins=np.linspace(0.0, 1.0, 21),
    
            # KDE
            draw_kde=True,
            kde_grid=np.linspace(0.0, 1.0, 100, endpoint=False),
            kappa_kde=10.0,
    
            # uniformity test
            do_uniformity_tests=1,
            tests=("h"),
            n_null=100000,
            mmax_h=20,
            nbins=8,
    
            # output
            save_dir=None,
            file_tag="",
            rc_dict=None
        )
    
        print_uniformity_summary(result)
        # plt.close()
        