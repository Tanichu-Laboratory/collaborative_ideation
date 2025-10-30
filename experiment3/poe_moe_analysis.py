#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter, NullLocator


# Utilities
def _ensure_prob(x, axis=None, eps=1e-12):
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    s = x.sum(axis=axis, keepdims=True)
    s = np.where(s <= 0, eps, s)
    return x / s

def js_divergence(p, q, grid=None, eps=1e-12):
    p = _ensure_prob(p)
    q = _ensure_prob(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return 0.5 * (kl_pm + kl_qm)

def nll(h_samples, P_h, H_grid, eps=1e-12):
    if len(h_samples) == 0: return float("nan")
    P_h = _ensure_prob(P_h)
    N = len(H_grid)
    idx = np.clip(np.round(h_samples * (N - 1)).astype(int), 0, N - 1)
    p = np.maximum(P_h[idx], eps)
    return float(-np.mean(np.log(p)))

def gauss_circ_1d(H, h0, sig):
    if sig <= 0: sig = 1e-3
    d = np.minimum(np.abs(H - h0), 1.0 - np.abs(H - h0))
    y = np.exp(-0.5 * (d / sig) ** 2)
    return _ensure_prob(y)

def boltzmann_dist_2d(utility_grid, beta=1.0):
    if beta <= 0: return _ensure_prob(np.ones_like(utility_grid))
    U = np.asarray(utility_grid, dtype=float)
    exp_u = np.exp(beta * (U - np.max(U))) # For numerical stability
    return _ensure_prob(exp_u)    

def kde_hc_2d(hc_list, H, C, bw_h, bw_c):
    H, C = np.asarray(H), np.asarray(C)
    Z = np.zeros((H.size, C.size), dtype=float)
    if len(hc_list) == 0: return _ensure_prob(np.ones_like(Z))
    bw_h, bw_c = max(1e-4, float(bw_h)), max(1e-4, float(bw_c))
    for h, c in hc_list:
        dh = np.minimum(np.abs(H - h), 1.0 - np.abs(H - h))
        Kh = np.exp(-0.5 * (dh / bw_h) ** 2)
        Kc = np.exp(-0.5 * ((C - c) / bw_c) ** 2)
        Z += Kh[:, None] * Kc[None, :]
    return _ensure_prob(Z)

def pref_gauss_2d(H, C, mu_h, sig_h, mu_c, sig_c, c_min=None, c_max=None):
    Gh = gauss_circ_1d(H, mu_h, sig_h)
    if sig_c <= 0: sig_c = 1e-3
    Gc = np.exp(-0.5 * ((C - mu_c) / sig_c) ** 2)
    L2 = Gh[:, None] * Gc[None, :]
    if c_min is not None or c_max is not None:
        mask = np.ones_like(L2, dtype=bool)
        if c_min is not None: mask &= (C[None, :] >= float(c_min))
        if c_max is not None: mask &= (C[None, :] <= float(c_max))
        L2 = np.where(mask, L2, 0.0)
    return _ensure_prob(L2)

def marginalize_h(P2):
    if P2 is None: return None
    return _ensure_prob(P2.sum(axis=1))

def model_poe_gen(Ps2, L2, Pt2, wE=1.0, wS=1.0, wP=1.0, eps=1e-12):
    E = np.clip(L2, eps, None)
    S = np.clip(Ps2, eps, None)
    P = np.clip(Pt2, eps, None)
    G = (E**float(wE)) * (S**float(wS)) * (P**float(wP))
    return _ensure_prob(G)

def smooth_circular(y, bw):
    if y is None or bw <= 0: return y
    n = len(y)
    sigma_steps = max(1e-6, bw * n)
    k_idx = np.arange(n)
    d = np.minimum(k_idx, n - k_idx)
    k = np.exp(-0.5 * (d / sigma_steps) ** 2)
    k /= k.sum()
    ys = np.fft.ifft(np.fft.fft(y) * np.fft.fft(k)).real
    return _ensure_prob(np.clip(ys, 0, None))

def estimate_evaluator_boltzmann_from_csv(csv_path, H, C, bw_h, bw_c, beta_L):
    print(f"[info] Estimating Evaluator distribution from utility scores in {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise SystemExit(f"[error] CSV file not found: {csv_path}")

    if 'conf' not in df.columns or df.empty:
        print("[warning] No 'conf' scores found in CSV. Returning uniform distribution.")
        return _ensure_prob(np.ones((len(H), len(C))))

    acc = np.zeros((len(H), len(C)), dtype=float)
    for _, row in df.iterrows():
        acc += row['conf'] * kde_hc_2d([(row['h'], row['C'])], H, C, bw_h, bw_c)

    # --- log-density → Boltzmann---
    density_E = _ensure_prob(acc)                          # p_E(h,c)
    U_E = np.log(np.maximum(density_E, 1e-300))            # U = log p_E
    L2 = boltzmann_dist_2d(U_E, beta=beta_L)               # ∝ exp(β·U) = p_E^β

    return L2


COLOR_PAT = re.compile(r"(?:LCh\s*)?Color\s*List\s*:\s*\[([^\]]+)\]", re.I)

def parse_hc_from_msg(text):
    if not text: return []
    m = COLOR_PAT.search(text)
    body = m.group(1) if m else text
    tuples = re.findall(r"\(([^)]+)\)", body)
    out = []
    for tup in tuples:
        try:
            parts = [p.strip().replace("°", "") for p in tup.split(",")]
            nums = [float(p) for p in parts if p]
            if len(nums) >= 2:
                h, c = nums[1], nums[2] if len(nums) == 3 else nums[1]
                h_norm = (h % 360.0) / 360.0 if h > 1.0 else h % 1.0
                out.append((h_norm, np.clip(c, 0.0, 1.0)))
        except (ValueError, IndexError):
            continue
    return out

def load_rounds(path):
    with open(path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    rounds = {}
    for row in logs:
        r = int(row.get("round", -1))
        if r < 0: continue
        role, msg = (row.get("role") or "").strip(), row.get("msg") or ""
        if role:
            rounds.setdefault(r, {})[role] = msg
    return rounds

# Analysis functions
def create_analysis_record(
    H, t, path, Ps_h, L_h, Pt_h, Ptp_h,
    PoE3_h, MoE_full_h, MoE_evalprod_h,
    hc_tp, MoPoE_h=None,
    PoE3_raw_h=None, MoE_evalprod_raw_h=None, MoPoE_raw_h=None,
    **kwargs
):
    _poe_for_metric   = PoE3_raw_h if PoE3_raw_h is not None else PoE3_h
    _moe_for_metric   = MoE_evalprod_raw_h if MoE_evalprod_raw_h is not None else MoE_evalprod_h
    _mopoe_for_metric = MoPoE_raw_h if MoPoE_raw_h is not None else MoPoE_h

    js_poe3  = js_divergence(_poe_for_metric,    Ptp_h) if (_poe_for_metric   is not None and Ptp_h is not None) else float('nan')
    js_moe_full     = js_divergence(MoE_full_h,  Ptp_h) if (MoE_full_h        is not None and Ptp_h is not None) else float('nan')
    js_moe_evalprod = js_divergence(_moe_for_metric, Ptp_h) if (_moe_for_metric is not None and Ptp_h is not None) else float('nan')
    js_mopoe        = js_divergence(_mopoe_for_metric,  Ptp_h) if (_mopoe_for_metric is not None and Ptp_h is not None) else float('nan')

    hs_tp = [h for h, c in hc_tp]
    nll_poe      = nll(hs_tp, _poe_for_metric, H)   if _poe_for_metric   is not None else float('nan')
    nll_moe      = nll(hs_tp, MoE_full_h, H)        if MoE_full_h        is not None else float('nan')
    nll_moe_prod = nll(hs_tp, _moe_for_metric, H)   if _moe_for_metric   is not None else float('nan')
    nll_mopoe    = nll(hs_tp, _mopoe_for_metric, H) if _mopoe_for_metric is not None else float('nan')

    return {
        "trial": os.path.basename(path), "round": t,
        "H": list(H),
        "Ps":  list(Ps_h), "L":   list(L_h),
        "Pt":  (list(Pt_h)  if Pt_h  is not None else None),
        "Ptp": (list(Ptp_h) if Ptp_h is not None else None),

        "PoE3":        (list(PoE3_h)        if PoE3_h        is not None else None),
        "MoE_full":    (list(MoE_full_h)    if MoE_full_h    is not None else None),
        "MoE_evalprod":(list(MoE_evalprod_h)if MoE_evalprod_h is not None else None),
        "MoPoE":       (list(MoPoE_h)       if MoPoE_h       is not None else None),

        "PoE3_raw":        (list(PoE3_raw_h)        if PoE3_raw_h        is not None else None),
        "MoE_evalprod_raw":(list(MoE_evalprod_raw_h)if MoE_evalprod_raw_h is not None else None),
        "MoPoE_raw":       (list(MoPoE_raw_h)       if MoPoE_raw_h       is not None else None),

        "js_poe": js_poe3, "js_moe": js_moe_full, "js_moe_prod": js_moe_evalprod, "js_mopoe": js_mopoe,
        "nll_poe": nll_poe, "nll_moe": nll_moe, "nll_moe_prod": nll_moe_prod, "nll_mopoe": nll_mopoe,

        **kwargs
    }


def analyze_one_trial_2d(path, H, C, L2, Ps2, args):
    rounds = load_rounds(path)
    if recursive := args.recursive:
        print(f"[info] Running in RECURSIVE mode for {os.path.basename(path)}.")
        priors = {'poe': Ps2.copy(), 'moe_full': Ps2.copy(), 'moe_prod': Ps2.copy()}

        if args.uniform_p0:
            uni = _ensure_prob(np.ones_like(Ps2))
            priors = {'poe': uni.copy(), 'moe_full': uni.copy(), 'moe_prod': uni.copy()}
        else:
            priors = {'poe': Ps2.copy(), 'moe_full': Ps2.copy(), 'moe_prod': Ps2.copy()}

    recs = []

    for t in sorted(rounds):
        hc_t = parse_hc_from_msg(rounds.get(t, {}).get('Evaluator', ''))
        Pt2  = kde_hc_2d(hc_t, H, C, args.bw_h, args.bw_c)
        if (t + 1) in rounds:
            hc_tp = parse_hc_from_msg(rounds.get(t + 1, {}).get('Evaluator', ''))
            Ptp2  = kde_hc_2d(hc_tp, H, C, args.bw_h, args.bw_c)
        else:
            hc_tp = []
            Ptp2  = None


        prior_poe = priors['poe'] if recursive else Pt2
        prior_moe_full = priors['moe_full'] if recursive else Pt2
        prior_moe_prod = priors['moe_prod'] if recursive else Pt2

        Ppoe3_2 = model_poe_gen(Ps2, L2, prior_poe, wE=args.wE_poe, wS=args.wS_poe, wP=args.wP_poe)
        cond_full = _ensure_prob(args.beta * Ps2 + (1.0 - args.beta) * prior_moe_full)
        Pmoe_full2 = _ensure_prob(args.wE * L2 + (1.0 - args.wE) * cond_full)
        cond_prod = _ensure_prob(args.beta * Ps2 + (1.0 - args.beta) * prior_moe_prod)
        Pmoe_evalprod2 = _ensure_prob(L2 * cond_prod)
        Pmopoe2 = _ensure_prob(args.alpha_mopoe * Ppoe3_2 + (1.0 - args.alpha_mopoe) * Pmoe_evalprod2)


        if recursive:
            priors.update({'poe': Ppoe3_2, 'moe_full': Pmoe_full2, 'moe_prod': Pmoe_evalprod2})
            
        raw_Pt_h    = marginalize_h(Pt2)
        raw_Ptp_h   = marginalize_h(Ptp2) if Ptp2 is not None else None

        raw_PoE_h   = marginalize_h(Ppoe3_2)
        raw_MoE_h   = marginalize_h(Pmoe_evalprod2)
        raw_MoPoE_h = marginalize_h(Pmopoe2)

        sm_PoE_h     = smooth_circular(raw_PoE_h, args.bw_h)
        sm_MoEfull_h = smooth_circular(marginalize_h(Pmoe_full2), args.bw_h)
        sm_MoE_h     = smooth_circular(raw_MoE_h, args.bw_h)
        sm_MoPoE_h   = smooth_circular(raw_MoPoE_h, args.bw_h)

        recs.append(create_analysis_record(
            H, t, path,
            marginalize_h(Ps2), marginalize_h(L2),
            raw_Pt_h, raw_Ptp_h,        
            sm_PoE_h, sm_MoEfull_h, sm_MoE_h,  
            hc_tp,
            MoPoE_h=sm_MoPoE_h,
            PoE3_raw_h=raw_PoE_h, MoE_evalprod_raw_h=raw_MoE_h, MoPoE_raw_h=raw_MoPoE_h,
            mode="per-trial"
        ))

    return recs

def analyze_across_trials_2d(log_paths, H, C, L2, Ps2, args):
    print("[info] Running in standard AGGREGATE mode.")
    recs, emp_data = [], {'Pt': {}, 'Ptp': {}, 'hc_tp': {}}
    for path in log_paths:
        rounds = load_rounds(path)
        for t in rounds:
            if t + 1 in rounds:
                emp_data['Pt'].setdefault(t, []).extend(parse_hc_from_msg(rounds[t].get('Evaluator', '')))
                emp_data['Ptp'].setdefault(t + 1, []).extend(parse_hc_from_msg(rounds[t+1].get('Evaluator', '')))
                emp_data['hc_tp'].setdefault(t+1, []).extend(parse_hc_from_msg(rounds[t+1].get('Evaluator', '')))

    for t in sorted(emp_data['Pt'].keys()):
        Pt2 = kde_hc_2d(emp_data['Pt'][t], H, C, args.bw_h, args.bw_c)
        Ptp2 = kde_hc_2d(emp_data['Ptp'][t+1], H, C, args.bw_h, args.bw_c)

        Ppoe3_2 = model_poe_gen(Ps2, L2, Pt2, wE=args.wE_poe, wS=args.wS_poe, wP=args.wP_poe)
        cond_full = _ensure_prob(args.beta * Ps2 + (1.0 - args.beta) * Pt2)
        Pmoe_full2 = _ensure_prob(args.wE * L2 + (1.0 - args.wE) * cond_full)
        cond_prod = _ensure_prob(args.beta * Ps2 + (1.0 - args.beta) * Pt2)
        Pmoe_evalprod2 = _ensure_prob(L2 * cond_prod)
        Pmopoe2 = _ensure_prob(args.alpha_mopoe * Ppoe3_2 + (1.0 - args.alpha_mopoe) * Pmoe_evalprod2)

        raw_Pt_h    = marginalize_h(Pt2)
        raw_Ptp_h   = marginalize_h(Ptp2)
        raw_PoE_h   = marginalize_h(Ppoe3_2)
        raw_MoE_h   = marginalize_h(Pmoe_evalprod2)
        raw_MoPoE_h = marginalize_h(Pmopoe2)

        sm_PoE_h     = smooth_circular(raw_PoE_h, args.bw_h)
        sm_MoEfull_h = smooth_circular(marginalize_h(Pmoe_full2), args.bw_h)
        sm_MoE_h     = smooth_circular(raw_MoE_h, args.bw_h)
        sm_MoPoE_h   = smooth_circular(raw_MoPoE_h, args.bw_h)

        recs.append(create_analysis_record(
            H, t, "aggregate",
            marginalize_h(Ps2),  marginalize_h(L2),
            raw_Pt_h,            raw_Ptp_h,            
            sm_PoE_h,            sm_MoEfull_h,         
            sm_MoE_h,
            emp_data['hc_tp'][t+1],
            MoPoE_h=sm_MoPoE_h,
            PoE3_raw_h=raw_PoE_h,
            MoE_evalprod_raw_h=raw_MoE_h,
            MoPoE_raw_h=raw_MoPoE_h,
            mode="aggregate",
            n_t=len(emp_data['Pt'][t]), n_tp=len(emp_data['Ptp'][t+1])
        ))
    
    return recs

def analyze_aggregate_recursive(log_paths, H, C, L2, Ps2, args):

    print("[info] Running in AGGREGATE RECURSIVE (simulation) mode.")
    sim_results = {'PoE3': {}, 'MoE_full': {}, 'MoE_evalprod': {}, 'MoPoE': {}}

    emp_data = {'Pt': {}, 'Ptp': {}, 'hc_tp': {}}
    
    for path in log_paths:
        rounds = load_rounds(path)
        Ts = sorted(r for r in rounds if r+1 in rounds)
        if not Ts: continue
        priors = {'poe': Ps2.copy(), 'moe_full': Ps2.copy(), 'moe_prod': Ps2.copy()}
        
        for t in Ts:
            pred_poe3 = model_poe_gen(Ps2, L2, priors['poe'], wE=args.wE_poe, wS=args.wS_poe, wP=args.wP_poe)
            sim_results['PoE3'].setdefault(t+1, []).append(pred_poe3)
            priors['poe'] = pred_poe3
            
            cond_full = _ensure_prob(args.beta * Ps2 + (1.0 - args.beta) * priors['moe_full'])
            pred_moe_full = _ensure_prob(args.wE * L2 + (1.0 - args.wE) * cond_full)
            sim_results['MoE_full'].setdefault(t+1, []).append(pred_moe_full)
            priors['moe_full'] = pred_moe_full
            
            cond_prod = _ensure_prob(args.beta * Ps2 + (1.0 - args.beta) * priors['moe_prod'])
            pred_moe_prod = _ensure_prob(L2 * cond_prod)
            sim_results['MoE_evalprod'].setdefault(t+1, []).append(pred_moe_prod)
            priors['moe_prod'] = pred_moe_prod

            pred_mopoe = _ensure_prob(args.alpha_mopoe * pred_poe3 + (1.0 - args.alpha_mopoe) * pred_moe_prod)
            sim_results['MoPoE'].setdefault(t+1, []).append(pred_mopoe)


            emp_data['Pt'].setdefault(t, []).extend(parse_hc_from_msg(rounds[t].get('Evaluator', '')))
            emp_data['Ptp'].setdefault(t+1, []).extend(parse_hc_from_msg(rounds[t+1].get('Evaluator', '')))
            emp_data['hc_tp'].setdefault(t+1, []).extend(parse_hc_from_msg(rounds[t+1].get('Evaluator', '')))
            
    recs = []

    for t_pred in sorted(sim_results['PoE3'].keys()):
        t_obs = t_pred - 1
        avg_poe3_2d = np.mean(np.array(sim_results['PoE3'][t_pred]), axis=0)
        avg_moe_full_2d = np.mean(np.array(sim_results['MoE_full'][t_pred]), axis=0)
        avg_moe_prod_2d = np.mean(np.array(sim_results['MoE_evalprod'][t_pred]), axis=0)
        Pt2_agg = kde_hc_2d(emp_data['Pt'].get(t_obs, []), H, C, args.bw_h, args.bw_c)
        Ptp2_agg = kde_hc_2d(emp_data['Ptp'].get(t_pred, []), H, C, args.bw_h, args.bw_c)
        avg_mopoe_2d = _ensure_prob(np.mean(np.array(sim_results['MoPoE'][t_pred]), axis=0))

        raw_Pt_h    = marginalize_h(Pt2_agg)
        raw_Ptp_h   = marginalize_h(Ptp2_agg)
        raw_PoE_h   = marginalize_h(avg_poe3_2d)
        raw_MoE_h   = marginalize_h(avg_moe_prod_2d)
        raw_MoPoE_h = marginalize_h(avg_mopoe_2d)

        sm_PoE_h     = smooth_circular(raw_PoE_h, args.bw_h)
        sm_MoEfull_h = smooth_circular(marginalize_h(avg_moe_full_2d), args.bw_h)
        sm_MoE_h     = smooth_circular(raw_MoE_h, args.bw_h)
        sm_MoPoE_h   = smooth_circular(raw_MoPoE_h, args.bw_h)

        recs.append(create_analysis_record(
            H, t_obs, "agg_recursive",
            marginalize_h(Ps2),  marginalize_h(L2),
            raw_Pt_h,            raw_Ptp_h,      
            sm_PoE_h,            sm_MoEfull_h,   
            sm_MoE_h,
            emp_data['hc_tp'][t_pred],
            MoPoE_h=sm_MoPoE_h,
            PoE3_raw_h=raw_PoE_h,
            MoE_evalprod_raw_h=raw_MoE_h,
            MoPoE_raw_h=raw_MoPoE_h,
            mode="agg_recursive",
            n_t=len(emp_data['Pt'].get(t_obs, [])),
            n_tp=len(emp_data['Ptp'].get(t_pred, []))
        ))


        
    return recs

def plot_small_multiples(records, out_png, title_prefix="", picks=None, alpha_mopoe=None):

    rec_by_round = {}
    for r in records:
        rec_by_round.setdefault(r["round"], []).append(r)
    rounds = sorted(rec_by_round.keys())
    if not rounds:
        return

    if picks is None:
        picks = [rounds[0], rounds[len(rounds)//2], rounds[-1]]

    fig, axes = plt.subplots(1, len(picks), figsize=(4.0*len(picks), 3.4), sharey=True, squeeze=False)
    for i, t in enumerate(picks):
        ax = axes[0, i]
        r = rec_by_round[t][0]
        H = np.array(r["H"])
        ax.plot(H, r["Pt"],  label="Emp(t)",   lw=2)
        if r.get("Ptp") is not None:                    
            ax.plot(H, r["Ptp"], label="Emp(t+1)", lw=2)
        if r.get("PoE3"): ax.plot(H, r["PoE3"], label="PoE", lw=2)
        if r.get("MoE_evalprod"): ax.plot(H, r["MoE_evalprod"], label="MoE", lw=2)
        if r.get("MoPoE") is not None:
            ax.plot(H, r["MoPoE"], label=f"MoPoE (α={alpha_mopoe})", lw=2)
        if r.get("L"): ax.plot(H, r["L"], label="Eval. Preference", lw=2, ls=":", color="black", alpha=0.7)
        if r.get("Ps"): ax.plot(H, r["Ps"], label="Samp. Prior", lw=2, ls="--", color="gray", alpha=0.8)
        ax.set_title(f"round {t}")
        ax.set_xlabel("Hue (wrapped 0..1)")
        ax.grid(alpha=0.15)
    
    axes[0,0].set_ylabel("Density")
    h, l = axes[0,0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.88, 0.5))
    if title_prefix: fig.suptitle(title_prefix, y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_agg_small_multiples(agg_recs, out_png, title_prefix="agg (pooled across trials)", alpha_mopoe=None):

    PALETTE = {
        "poe": "skyblue",
        "moe": "lightgreen",
        "mopoe": "plum",
        "emp_t1": "salmon",
        "sampler": "gray",
        "evaluator": "black",
    }


    by_round = {r["round"]: r for r in agg_recs}
    rounds = sorted(by_round.keys())
    if not rounds: return
    picks = [rounds[0], rounds[len(rounds)//2], rounds[-1]] 
    fig, axes = plt.subplots(1, len(picks), figsize=(4.6*len(picks), 3.8),
                             sharey=True, squeeze=False, gridspec_kw={"wspace": 0.12})

    sublabels = ["(A)", "(B)", "(C)"]  

    for i, t in enumerate(picks):
        ax = axes[0, i]
        r = by_round[t]
        H = np.array(r["H"])
        
        ax.plot(H, r["Pt"],  label="Emp(t)",   lw=2, color=PALETTE["emp_t1"], alpha=0.4) 
        if r.get("Ptp") is not None:                     
            ax.plot(H, r["Ptp"], label="Emp(t+1)", lw=2, color=PALETTE["emp_t1"])
        if r.get("PoE3"): ax.plot(H, r["PoE3"], label="PoE", lw=2, color=PALETTE["poe"])
        if r.get("MoE_evalprod"): ax.plot(H, r["MoE_evalprod"], label="MoE", lw=2, color=PALETTE["moe"])
        if r.get("MoPoE") is not None:
            ax.plot(H, r["MoPoE"], label=f"MoPoE", lw=2, color=PALETTE["mopoe"])
        if r.get("L"): ax.plot(H, r["L"], label="Eval. Preference", lw=2, ls=":", color=PALETTE["evaluator"], alpha=0.4)
        if r.get("Ps"): ax.plot(H, r["Ps"], label="Samp. Prior", lw=2, ls="--", color=PALETTE["sampler"], alpha=0.4)

        ax.set_xlabel("Hue (wrapped 0..1)", fontsize=20, labelpad=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(NullLocator()) 
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.text(0.5, -0.26, sublabels[i], transform=ax.transAxes,
                ha="center", va="top", fontsize=18)
        ax.grid(alpha=0.15)

    axes[0,0].set_ylabel("Density", fontsize=20, labelpad=4)
    ax.set_ylim(0.0000, 0.0125)
    h, l = axes[0,0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.87, 0.57), fontsize=20)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.95, bottom=0.18, wspace=0.30)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tiny_curve_png(H, y, path, color="#d95b5b", ymax=None, lw=1.0):
    import numpy as np, matplotlib.pyplot as plt
    H = np.asarray(H); y = np.asarray(y, dtype=float)
    y = y / (y.sum() if y.sum() > 0 else 1.0)
    if ymax is None: ymax = float(y.max()) * 1.05

    w = 1.2  
    fig, ax = plt.subplots(figsize=(w, 0.3), dpi=500)
    ax.plot(H, y, color=color, lw=lw)
    ax.axhline(0, color=(0,0,0,0.12), lw=0.8)   
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0, ymax)
    ax.set_axis_off(); plt.margins(0, 0)
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="2D PoE/MoE analysis for LCh hue-chroma.")
    ap.add_argument("--logs", nargs="+", required=True, help="Glob(s) for trial*.json")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory")
    ap.add_argument("--outcsv", default="metrics.csv", help="CSV filename")
    ap.add_argument("--plot", action="store_true", help="Save small-multiples figure for per-trial")
    ap.add_argument("--fig",  default="small_multiples.png", help="Per-trial figure filename")
    ap.add_argument("--aggregate", action="store_true", help="Pool across trials per round")
    ap.add_argument("--uniform_p0", action="store_true",
                    help="When --recursive, initialize priors with uniform p0 instead of Ps2.")
    ap.add_argument("--agg_fig", default="agg_small_multiples.png", help="Aggregate figure filename")
    ap.add_argument("--grid", type=int, default=512, help="Hue grid size (0..1)")
    ap.add_argument("--bw_h", type=float, default=0.06, help="KDE bandwidth on hue (0..1)")
    ap.add_argument("--bw_c", type=float, default=0.06, help="KDE bandwidth on chroma (0..1)")
    ap.add_argument("--c_min", type=float, default=0.0)
    ap.add_argument("--c_max", type=float, default=1.0)
    ap.add_argument("--wE_poe", type=float, default=1.0)
    ap.add_argument("--wS_poe", type=float, default=1.0)
    ap.add_argument("--wP_poe", type=float, default=1.0)
    ap.add_argument("--wE",    type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--alpha_mopoe", type=float, default=0.5,
                    help="MoPoE mixing weight alpha in P_mopoe = alpha*PoE + (1-alpha)*MoE_evalprod")
    ap.add_argument("--beta_L", type=float, default=1.0, help="Inverse temperature (beta) for Evaluator's preference distribution from scores.")
    ap.add_argument("--beta_S", type=float, default=1.0, help="Inverse temperature (beta) for Sampler's prior distribution.")
    ap.add_argument("--recursive", action="store_true", help="Run in recursive prediction (simulation) mode.")
    ap.add_argument("--evaluator_dist_csv", type=str, required=True, help="CSV file with h, C, conf columns to estimate Evaluator preference distribution.")
    ap.add_argument("--smooth_poe_moe", action="store_true", help="Plot smoothed model curves.")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    def _stamp_png(path):
        base, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}{ext}" if ext.lower() == ".png" else path

    paths = sorted([p for pat in args.logs for p in glob.glob(pat)])
    if not paths: raise SystemExit("No log files matched by --logs.")

    H = np.linspace(0.0, 1.0, args.grid, endpoint=True)
    C = np.linspace(args.c_min, args.c_max, max(16, args.grid//4), endpoint=True)
    L2 = estimate_evaluator_boltzmann_from_csv(args.evaluator_dist_csv, H, C, args.bw_h, args.bw_c, args.beta_L)

    print("[info] Estimating Sampler's prior distribution from Round 1 samples...")
    sampler_r1_samples = []
    for p in paths:
        rounds = load_rounds(p)
        if r1_data := rounds.get(1):
            if sampler_msg := r1_data.get('Sampler'):
                sampler_r1_samples.extend(parse_hc_from_msg(sampler_msg))
    
    sampler_density_grid = kde_hc_2d(sampler_r1_samples, H, C, args.bw_h, args.bw_c)  # p_S(h,c)
    U_S = np.log(np.maximum(sampler_density_grid, 1e-300))                            # U = log p_S


    # ===== Debug =====
    print("[DEBUG] Saving utility grid heatmap to utility_grid_S.png")
    fig, ax = plt.subplots()
    im = ax.imshow(U_S.T, origin='lower', aspect='auto',
               extent=[H.min(), H.max(), C.min(), C.max()])
    ax.set_title("Utility (U_S = log p_S)")
    ax.set_xlabel("Hue (h)")
    ax.set_ylabel("Chroma (C)")
    fig.colorbar(im, ax=ax)
    fig.savefig("utility_grid_S.png")
    plt.close(fig)
    # ===== Debug =====

    Ps2 = boltzmann_dist_2d(U_S, beta=args.beta_S)                                    

    pe_h = marginalize_h(L2)   # Evaluator prior over hue
    ps_h = marginalize_h(Ps2)  # Sampler prior over hue

    tiny_dir = os.path.join(args.outdir, "tiny_priors_exp3")
    os.makedirs(tiny_dir, exist_ok=True)

    shared_ymax = 1.05 * float(max(pe_h.max(), ps_h.max()))
    save_tiny_curve_png(H, pe_h, os.path.join(tiny_dir, "prior_PE.png"),
                        color="#d95b5b", ymax=shared_ymax)
    save_tiny_curve_png(H, ps_h, os.path.join(tiny_dir, "prior_PS.png"),
                        color="#4477aa", ymax=shared_ymax)

    if args.aggregate:
        if args.recursive:
            recs = analyze_aggregate_recursive(paths, H, C, L2, Ps2, args)
            title = "agg_recursive (pooled & simulated)"
        else:
            recs = analyze_across_trials_2d(paths, H, C, L2, Ps2, args)

            title = "agg (pooled)"
        df = pd.DataFrame(recs)
        df.to_csv(os.path.join(args.outdir, f"{title.split(' ')[0]}_{args.outcsv}"), index=False)
        fig_path = _stamp_png(os.path.join(args.outdir, args.agg_fig))
        plot_agg_small_multiples(recs, fig_path, alpha_mopoe=args.alpha_mopoe)


    else:
        all_recs = []
        for p in paths:
            recs = analyze_one_trial_2d(p, H, C, L2, Ps2, args)
            all_recs.extend(recs)
            if args.plot and recs:
                base = os.path.splitext(os.path.basename(p))[0]
                fig_name = f"{base}_recursive_{args.fig}" if args.recursive else f"{base}_{args.fig}"
                fig_path = _stamp_png(os.path.join(args.outdir, fig_name))
                title = f"{base} (recursive)" if args.recursive else base
                plot_small_multiples(recs, fig_path, title_prefix=title, alpha_mopoe=args.alpha_mopoe)

        if all_recs:
            df = pd.DataFrame(all_recs)
            csv_name = "recursive_" + args.outcsv if args.recursive else args.outcsv
            df.to_csv(os.path.join(args.outdir, csv_name), index=False)

if __name__ == "__main__":
    main()