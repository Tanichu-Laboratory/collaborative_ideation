#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# --- Constants & Utilities ---
COUNTRY_LIST = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
    "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe"
]
COUNTRY_INDEX = {name: i for i, name in enumerate(COUNTRY_LIST)}
NUM_COUNTRIES = len(COUNTRY_LIST)

def boltzmann_selection(utilities, temperature=1.0):
    utilities = np.array(utilities)
    stable_utilities = utilities / temperature - np.max(utilities / temperature)
    exp_utilities = np.exp(stable_utilities)
    return exp_utilities / np.sum(exp_utilities)

def _ensure_prob(x, eps=1e-12):
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    s = x.sum()
    return x / (s if s > 0 else eps)

def js_divergence(p, q, eps=1e-12):
    p, q = _ensure_prob(p), _ensure_prob(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return 0.5 * (kl_pm + kl_qm)

def nll(samples: list[str], prob_dist: np.ndarray) -> float:
    if not samples: return float("nan")
    prob_dist = _ensure_prob(prob_dist)
    indices = [COUNTRY_INDEX[name] for name in samples if name in COUNTRY_INDEX]
    if not indices: return float("nan")
    probabilities = np.maximum(prob_dist[indices], 1e-12)
    return float(-np.mean(np.log(probabilities)))

def model_poe(p_sampler, p_evaluator, p_current):
    p_next = p_current * p_sampler * p_evaluator
    return _ensure_prob(p_next)

def model_moe_evalprod(p_sampler, p_evaluator, p_current, beta):
    mixed_dist = beta * p_sampler + (1 - beta) * p_current
    p_next = p_evaluator * mixed_dist
    return _ensure_prob(p_next)

COUNTRY_PAT = re.compile(r"Country List:\s*\[([^\]]+)\]", re.I)

def parse_countries_from_msg(text: str) -> list[str]:
    m = COUNTRY_PAT.search(text or "")
    if not m: return []
    countries = [item.strip().strip("'\"") for item in m.group(1).split(",")]
    return [country for country in countries if country]

def calculate_histogram_dist(samples: list[str]) -> np.ndarray:
    if not samples:
        return np.ones(NUM_COUNTRIES) / NUM_COUNTRIES
    counts = np.zeros(NUM_COUNTRIES)
    for country in samples:
        if country in COUNTRY_INDEX:
            counts[COUNTRY_INDEX[country]] += 1
    return _ensure_prob(counts)

def analyze_logs(all_logs_data, p_sampler, p_evaluator, beta, alpha):
    empirical_data = {}
    for trial_events in all_logs_data:
        for entry in trial_events:
            r = entry.get('round')
            role = entry.get('role')
            if r is None or role not in ["Sampler", "Evaluator"]:
                continue
            empirical_data.setdefault(r, {"Sampler": [], "Evaluator": []})
            empirical_data[r][role].extend(parse_countries_from_msg(entry.get("msg", "")))

    sorted_rounds = sorted(empirical_data.keys())
    records = []

    # Initialize theoretical belief states to uniform (belief at Round 0)
    p_belief_poe = np.ones(NUM_COUNTRIES) / NUM_COUNTRIES
    p_belief_moe = np.ones(NUM_COUNTRIES) / NUM_COUNTRIES

    # Loop through rounds to simulate theoretical evolution and compare to empirical data
    for r in sorted_rounds:
        # Store current beliefs for the next iteration's debug print
        p_current_poe_for_debug = np.copy(p_belief_poe)
        p_current_moe_for_debug = np.copy(p_belief_moe)

        p_belief_poe = model_poe(p_sampler, p_evaluator, p_belief_poe)
        p_belief_moe = model_moe_evalprod(p_sampler, p_evaluator, p_belief_moe, beta)
        p_belief_mopoe = alpha * p_belief_poe + (1 - alpha) * p_belief_moe
        
        p_emp_r = calculate_histogram_dist(empirical_data[r]["Evaluator"])
        samples_r = empirical_data[r]["Evaluator"]

        # Calculate JSD and NLL for the updated state against the empirical data
        js_poe = js_divergence(p_belief_poe, p_emp_r)
        nll_poe = nll(samples_r, p_belief_poe)
        
        js_moe = js_divergence(p_belief_moe, p_emp_r)
        nll_moe = nll(samples_r, p_belief_moe)
        
        js_mopoe = js_divergence(p_belief_mopoe, p_emp_r)
        nll_mopoe = nll(samples_r, p_belief_mopoe)

        # Baselines are compared against the same empirical data
        js_sampler_only = js_divergence(p_sampler, p_emp_r)
        nll_sampler_only = nll(samples_r, p_sampler)
        js_evaluator_only = js_divergence(p_evaluator, p_emp_r)
        nll_evaluator_only = nll(samples_r, p_evaluator)

        records.append({
            "round": r, 
            "js_poe": js_poe, "js_moe": js_moe, "js_mopoe": js_mopoe,
            "nll_poe": nll_poe, "nll_moe": nll_moe, "nll_mopoe": nll_mopoe, 
            "js_sampler_only": js_sampler_only,
            "nll_sampler_only": nll_sampler_only,
            "js_evaluator_only": js_evaluator_only,
            "nll_evaluator_only": nll_evaluator_only,            
            "p_emp_t_plus_1": p_emp_r, 
            "p_pred_poe": p_belief_poe,
            "p_pred_moe": p_belief_moe,
            "p_pred_mopoe": p_belief_mopoe
        })
        
    return pd.DataFrame(records)

def plot_histograms(df: pd.DataFrame, p_sampler: np.ndarray, p_evaluator: np.ndarray, output_dir: str, scenario_id=None, is_unbiased=False):
    """Plots comparison histograms for select rounds with a unified X and Y axis across all figures."""
    if df.empty:
        print("Not enough data to plot.")
        return

    rounds_to_plot = [1, 5, 9]
    
    if not rounds_to_plot:
        print("No valid rounds with t and t+1 data found to plot.")
        return

    global_max_y = 0
    for r in rounds_to_plot:
        data = df[df['round'] == r].iloc[0]
        current_max = max(
            data['p_pred_poe'].max(), data['p_pred_moe'].max(),
            data['p_pred_mopoe'].max(), data['p_emp_t_plus_1'].max()
        )
        if current_max > global_max_y:
            global_max_y = current_max
    ylim_top = global_max_y * 1.15 

    combined_all_rounds = (df['p_pred_poe'].sum() + df['p_pred_moe'].sum() + 
                           df['p_pred_mopoe'].sum() + df['p_emp_t_plus_1'].sum())
    
    PROBABILITY_THRESHOLD = 0.20
    universal_indices = np.where(combined_all_rounds > PROBABILITY_THRESHOLD)[0]

    universal_labels = np.array(COUNTRY_LIST)[universal_indices]
    x = np.arange(len(universal_labels))
    p_sampler_filtered = p_sampler[universal_indices]
    p_evaluator_filtered = p_evaluator[universal_indices]

    # --- Helper function for plotting each subplot ---
    def _plot_subplot(ax, x_coords, pred_data, emp_data, title, pred_label, pred_color):
        width = 0.4
        # Main comparison bars
        ax.bar(x_coords - width/2, pred_data, width, label=pred_label, color=pred_color)
        ax.bar(x_coords + width/2, emp_data, width, label='Empirical at t+1', color='salmon')
        
        # Sampler Prior (stem plot)
        markerline, stemlines, baseline = ax.stem(
            x_coords, p_sampler_filtered, linefmt='grey', markerfmt='.', basefmt=' ', label='Sampler Prior'
        )
        plt.setp(stemlines, linestyle='--', alpha=0.7, linewidth=1)
        plt.setp(markerline, markersize=3)

        # Evaluator Prior (stem plot)
        markerline, stemlines, baseline = ax.stem(
            x_coords, p_evaluator_filtered, linefmt='black', markerfmt='.', basefmt=' ', label='Evaluator Prior'
        )
        plt.setp(stemlines, linestyle=':', alpha=0.8, linewidth=1.2)
        plt.setp(markerline, markersize=3)
        
        ax.set_ylabel('Probability', fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.legend(fontsize=18)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Generate plot for each selected round ---
    for r in rounds_to_plot:
        data = df[df['round'] == r].iloc[0]
        
        fig, axes = plt.subplots(3, 1, figsize=(max(15, len(universal_labels) * 0.4), 12), sharex=True, sharey=True)
        fig.suptitle(f"Distribution Comparison at Round {r} -> {r+1}", fontsize=22)
        axes[0].set_ylim(0, ylim_top)

        # Plot 1: PoE vs Empirical
        _plot_subplot(axes[0], x, data['p_pred_poe'][universal_indices], data['p_emp_t_plus_1'][universal_indices],
                      'PoE vs. Empirical Data', 'PoE Prediction for t+1', 'skyblue')

        # Plot 2: MoE vs Empirical
        _plot_subplot(axes[1], x, data['p_pred_moe'][universal_indices], data['p_emp_t_plus_1'][universal_indices],
                      'MoE vs. Empirical Data', 'MoE Prediction for t+1', 'lightgreen')

        # Plot 3: MoPoE vs Empirical
        _plot_subplot(axes[2], x, data['p_pred_mopoe'][universal_indices], data['p_emp_t_plus_1'][universal_indices],
                      'MoPoE vs. Empirical Data', 'MoPoE Prediction for t+1', 'plum')

        plt.xticks(x, universal_labels, rotation=90, fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        filename_parts = ["dist_comparison"]
        if scenario_id is not None:
            filename_parts.append(f"id{scenario_id}")
        
        condition_str = "unbiased" if is_unbiased else "biased"
        filename_parts.append(condition_str)
        filename_parts.append(f"round_{r}")

        filename = "_".join(filename_parts) + ".png"
        fig_path = os.path.join(output_dir, filename)


        plt.savefig(fig_path, dpi=150)
        print(f"Saved plot: {fig_path}")
        plt.close(fig)

def run_single_analysis(logs_dir, scores_csv, outdir, beta, alpha, tau_e, tau_s, scenario_id, is_unbiased):

    print("\n" + "="*80)
    print(f"===== Running Analysis for Scenario ID: {scenario_id} ({'Unbiased' if is_unbiased else 'Biased'}) =====")
    print("="*80)

    # --- Step 1: Create Evaluator Prior (p_evaluator) ---
    print(f"--- Step 1: Loading Evaluator scores from {os.path.basename(scores_csv)} ---")
    try:
        df_scores = pd.read_csv(scores_csv)
    except FileNotFoundError:
        print(f"Error: Evaluator scores file not found at {scores_csv}. Skipping scenario.")
        return

    mean_scores = df_scores.groupby('country')['score'].mean()
    evaluator_utilities = np.zeros(NUM_COUNTRIES)
    for i, country_name in enumerate(COUNTRY_LIST):
        evaluator_utilities[i] = mean_scores.get(country_name, 0.0)
    p_evaluator = boltzmann_selection(evaluator_utilities, temperature=tau_e)
    print(f"Successfully created p_evaluator using tau_e = {tau_e}.")

    # --- Step 2: Create Sampler Prior (p_sampler) ---
    print(f"--- Step 2: Loading logs from {os.path.basename(logs_dir)} ---")
    log_paths = glob.glob(os.path.join(logs_dir, "trial*.json"))
    if not log_paths:
        print(f"Error: No 'trial*.json' files found in '{logs_dir}'. Skipping scenario.")
        return
    
    sampler_round1_samples = []
    all_logs_data = []
    for path in log_paths:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                events = data.get("events", []) if isinstance(data, dict) else data
                all_logs_data.append(events)
                for entry in events:
                    if entry.get('round') == 1 and entry.get('role') == 'Sampler':
                        sampler_round1_samples.extend(parse_countries_from_msg(entry.get("msg", "")))
                        break
            except (json.JSONDecodeError, AttributeError):
                print(f"Warning: Could not parse {path}")

    counts = np.zeros(NUM_COUNTRIES, dtype=float)
    for name in sampler_round1_samples:
        idx = COUNTRY_INDEX.get(name)
        if idx is not None:
            counts[idx] += 1.0
    
    sampler_utilities = np.log1p(counts)
    p_sampler = boltzmann_selection(sampler_utilities, temperature=tau_s)
    print(f"Successfully created p_sampler using log transform and tau_s = {tau_s}.")
    
    # --- Generate tiny prior glyphs (PNG) for figure embedding ---
    glyph_dir = os.path.join(outdir, "tiny_priors")
    os.makedirs(glyph_dir, exist_ok=True)
    tag = f"id{scenario_id}_{'unbiased' if is_unbiased else 'biased'}"
    shared_ymax = 1.05 * max(float(p_evaluator.max()), float(p_sampler.max()))

    save_tiny_ranked_hist_png(
        p_evaluator,
        os.path.join(glyph_dir, f"{tag}_prior_PE.png"),
        color="#d95b5b",  
        topk=12,
        ymax=shared_ymax
    )
    save_tiny_ranked_hist_png(
        p_sampler,
        os.path.join(glyph_dir, f"{tag}_prior_PS.png"),
        color="#4477aa",  
        topk=12,
        ymax=shared_ymax
    )


    # --- Step 3: Run Analysis and Plotting ---
    print("--- Step 3: Running analysis and plotting ---")
    analysis_df = analyze_logs(all_logs_data, p_sampler, p_evaluator, beta, alpha)
    
    condition_str_for_filename = "unbiased" if is_unbiased else "biased"
    filename_parts = ["analysis_metrics", f"id{scenario_id}", condition_str_for_filename]
    filename = "_".join(filename_parts) + ".csv"
    csv_path = os.path.join(outdir, filename)

    columns_to_drop = ['p_emp_t_plus_1', 'p_pred_poe', 'p_pred_moe', 'p_pred_mopoe']
    analysis_df.drop(columns=columns_to_drop).to_csv(csv_path, index=False)
    print(f"Saved analysis metrics to: {csv_path}")

    plot_histograms(analysis_df, p_sampler, p_evaluator, outdir, scenario_id=scenario_id, is_unbiased=is_unbiased)
    print(f"===== Finished analysis for Scenario ID: {scenario_id} =====")


def save_tiny_hist_png(prior, path, color, topk=10, ymax=None):
    prior = np.asarray(prior, dtype=float)
    prior = prior / prior.sum()
    idx = np.argsort(prior)[::-1][:topk]
    vals = prior[idx]

    if ymax is None:
        ymax = vals.max() * 1.05

    w = max(0.9, 0.18 * topk)  
    fig, ax = plt.subplots(figsize=(w, 0.35), dpi=300)
    ax.bar(range(len(vals)), vals, width=0.8, color=color)
    ax.set_ylim(0, ymax)
    ax.set_axis_off(); plt.margins(0, 0)
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)

def save_tiny_ranked_hist_png(prior, path, color, topk=10, ymax=None):
    import numpy as np, matplotlib.pyplot as plt
    p = np.asarray(prior, dtype=float); p = p / p.sum()
    idx = np.argsort(p)[::-1]                
    top = p[idx][:topk]
    others = p[idx][topk:].sum()
    vals = np.r_[top, others]                
    if ymax is None: ymax = vals.max() * 1.05

    fig, ax = plt.subplots(figsize=(max(0.9, 0.18*len(vals)), 0.35), dpi=300)
    ax.bar(range(len(vals)), vals, width=0.8, color=color)
    ax.set_ylim(0, ymax); ax.set_axis_off(); plt.margins(0,0)
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run analysis for multiple travel brainstorming scenarios in a directory.")
    parser.add_argument("--root_dir", default=".", help="Path to the root directory containing 'travel_results_*' folders and 'preference_*_scores.csv' files. Defaults to the current directory.")
    parser.add_argument("--outdir", default="analysis_results_batch", help="Directory to save all analysis results.")    
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau_e", type=float, default=2.5)
    parser.add_argument("--tau_s", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    scenario_dirs = glob.glob(os.path.join(args.root_dir, "travel_results_*"))
    
    if not scenario_dirs:
        print(f"Error: No 'travel_results_*' directories found in '{args.root_dir}'.")
        return

    print(f"Found {len(scenario_dirs)} scenario directories to process.")

    for logs_dir in sorted(scenario_dirs):
        dir_name = os.path.basename(logs_dir)

        match = re.search(r'_id(\d+)_(biased|unbiased)_', dir_name)
        if not match:
            print(f"Warning: Could not parse scenario info from directory name '{dir_name}'. Skipping.")
            continue
            
        scenario_id = int(match.group(1))
        bias_str = match.group(2)
        is_unbiased = (bias_str == 'unbiased')
        
        scores_csv = os.path.join(args.root_dir, f"preference_{scenario_id}_scores.csv")
        
        run_single_analysis(
            logs_dir=logs_dir,
            scores_csv=scores_csv,
            outdir=args.outdir,
            beta=args.beta,
            alpha=args.alpha,
            tau_e=args.tau_e,
            tau_s=args.tau_s,
            scenario_id=scenario_id,
            is_unbiased=is_unbiased
        )
    
if __name__ == "__main__":
    main()