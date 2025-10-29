#!/usr/bin/env python3

import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import MultipleLocator, NullFormatter

# --- Utilities & Constants ---
SUITS = ['Heart', 'Diamond', 'Club', 'Spade']
RANKS = { '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 1 }
CARD_DECK_THEORY = [{'name': f"{suit} {rank_name}", 'suit': suit, 'rank': rank_val} for suit in SUITS for rank_name, rank_val in RANKS.items()]
CARD_NAMES_THEORY = [c['name'] for c in CARD_DECK_THEORY]
CARD_CODE_MAP = {f"{s[0]}{v}": f"{s} {k}" for s in SUITS for k, v in RANKS.items()}

#for plots
PALETTE = {
    "poe": "skyblue",
    "moe": "lightgreen",
    "mopoe": "plum",
    "emp_t1": "salmon",
    "sampler": "gray",
    "evaluator": "black",
}

SUIT_SYMBOL = {'D':'♦','H':'♥','C':'♣','S':'♠'}
def code_to_pretty(code: str) -> str:
    return f"{SUIT_SYMBOL.get(code[0], code[0])}{code[1:]}"


def get_sampler_utility(card: dict) -> float:
    """
    SAMPLER_INFO:
    - Suits ♦ (score: 5.4), ♣ (score: 5.1), ♥ (score: 4.7), ♠ (score: 4.6).
    - All ranks are treated similarly (score: 4.8).
    - Additive model: rank_prior + suit_prior
    """
    suit = card.get('suit')
    u = 0.0

    # rank: uniform prior
    u += 4.8

    # suit prior
    if suit == 'Diamond':
        u += 5.4
    elif suit == 'Club':
        u += 5.1
    elif suit == 'Heart':
        u += 4.7
    elif suit == 'Spade':
        u += 4.6
    else:
        u += 4.6  # fallback for unknown suit
    return u

def get_evaluator_utility(card: dict) -> float:
    """
    EVALUATOR_INFO:
    - Suit base relevance:
      - All suits share the same base (score: 1.0).

    - Rank bump (explicit priority tiers):
      - Tier A: ♦8 (score: +8.0)
      - Tier B: ♦7 (score: +3.5)
      - Tier C: ♦9 (score: +3.0)

    - Baseline consideration:
      - Other red cards (any ♦/♥ not above): (score: +0.4)
      - Other black cards (any ♠/♣ not above): (score: +0.2)
    """
    suit = card.get('suit')
    rank = int(card.get('rank'))
    u = 1.0  # uniform base for all suits

    # explicit tiers first
    if suit == 'Diamond' and rank == 8:
        return u + 8.0
    if suit == 'Diamond' and rank == 7:
        return u + 3.5
    if suit == 'Diamond' and rank == 9:
        return u + 3.0

    # otherwise apply baseline consideration
    if suit in ('Diamond', 'Heart'):  # other red cards
        u += 0.4
    else:  # black suits
        u += 0.2

    return u

def boltzmann_selection(utilities, temperature=1.0):
    utilities = np.array(utilities)
    stable_utilities = utilities / temperature - np.max(utilities / temperature)
    exp_utilities = np.exp(stable_utilities)
    return exp_utilities / np.sum(exp_utilities)

class CardExperimentSimulator:
    def __init__(self, temp_sampler, temp_evaluator, beta=0.5, alpha=0.9):
        self.card_names = CARD_NAMES_THEORY
        self.beta = beta
        self.alpha = alpha
        sampler_utilities = [get_sampler_utility(card) for card in CARD_DECK_THEORY]
        evaluator_utilities = [get_evaluator_utility(card) for card in CARD_DECK_THEORY]
        self.p_sampler = boltzmann_selection(sampler_utilities, temperature=temp_sampler)
        self.p_evaluator = boltzmann_selection(evaluator_utilities, temperature=temp_evaluator)
        self.p_base_poe = self.p_sampler * self.p_evaluator
        self.p_base_poe /= np.sum(self.p_base_poe)

    def run_simulation(self, target_card_name, rounds=10):
        p_belief_poe = np.ones(52) / 52.0
        p_belief_moe = np.ones(52) / 52.0
        history = []
        target_idx = self.card_names.index(target_card_name)

        for round_num in range(1, rounds + 1):
            p_belief_poe = p_belief_poe * self.p_base_poe
            p_belief_poe /= np.sum(p_belief_poe)
            mixed_dist = self.beta * self.p_sampler + (1 - self.beta) * p_belief_moe
            p_belief_moe = self.p_evaluator * mixed_dist
            p_belief_moe /= np.sum(p_belief_moe)
            
            p_belief_mopoe = self.alpha * p_belief_poe + (1 - self.alpha) * p_belief_moe

            history.append({
                'Round': round_num,
                'P(Target) - PoE': p_belief_poe[target_idx],
                'P(Target) - MoE-EvalProd': p_belief_moe[target_idx],
                'P(Target) - MoPoE': p_belief_mopoe[target_idx]
            })
        
        final_dist_mopoe = self.alpha * p_belief_poe + (1 - self.alpha) * p_belief_moe

        final_distributions = pd.DataFrame({
            'Card': self.card_names,
            'Final_P(PoE)': p_belief_poe,
            'Final_P(MoE-EvalProd)': p_belief_moe,
            'Final_P(MoPoE)': final_dist_mopoe
        }).sort_values(by='Final_P(PoE)', ascending=False).reset_index(drop=True)

        return pd.DataFrame(history), final_distributions

CARD_PAT_LOG = re.compile(r"Card List:\s*\[([^\]]+)\]", re.I)

def parse_cards_from_msg(text: str) -> list[str]:
    m = CARD_PAT_LOG.search(text or "")
    if not m: return []
    raw_items = [item.strip().strip("'\"") for item in m.group(1).split(",")]
    valid_cards = [item for item in raw_items if re.fullmatch(r"[SHCD](1[0-3]|[1-9])", item)]
    return valid_cards

def analyze_and_plot_final_distribution(all_final_winners, theoretical_final_dist_df, out_dir):
    print("\n--- Analysis 1: Final Choice Distribution (based on trial winners) ---")
    emp_counts = Counter(all_final_winners)
    total_trials = len(all_final_winners)
    
    top_n = 10
    top_cards_codes = [card for card, count in emp_counts.most_common(top_n)]
    top_cards_names = [CARD_CODE_MAP.get(code, code) for code in top_cards_codes]
    emp_probs = np.array([emp_counts.get(code, 0) / total_trials for code in top_cards_codes])
    
    theory_map_poe = dict(zip(theoretical_final_dist_df['Card'], theoretical_final_dist_df['Final_P(PoE)']))
    theory_map_moe = dict(zip(theoretical_final_dist_df['Card'], theoretical_final_dist_df['Final_P(MoE-EvalProd)']))
    theory_map_mopoe = dict(zip(theoretical_final_dist_df['Card'], theoretical_final_dist_df['Final_P(MoPoE)']))
    
    poe_probs = np.array([theory_map_poe.get(name, 0) for name in top_cards_names])
    moe_probs = np.array([theory_map_moe.get(name, 0) for name in top_cards_names])
    mopoe_probs = np.array([theory_map_mopoe.get(name, 0) for name in top_cards_names])
    
    x = np.arange(len(top_cards_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width*1.5, emp_probs, width, label=f'Empirical (N={total_trials} trials)', color='salmon')
    ax.bar(x - width*0.5, poe_probs, width, label='Theoretical PoE', color='skyblue')
    ax.bar(x + width*0.5, moe_probs, width, label='Theoretical MoE', color='lightgreen')
    ax.bar(x + width*1.5, mopoe_probs, width, label='Theoretical MoPoE', color='plum')
        
    ax.set_ylabel('Probability / Frequency', fontsize=22)
    ax.set_title('Final Choice Distribution: LLM Simulation vs. Theoretical Models', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(top_cards_names, ha="right", fontsize=18)

    ax.legend(fontsize=20)
    ax.tick_params(axis='y', labelsize=18) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    fig_path = out_dir / f"analysis_final_distribution_winners_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path)
    print(f"Saved final distribution plot to: {fig_path}")
    plt.close(fig)

def analyze_and_plot_roundly_success(all_logs, theoretical_history_df, target_card_code, out_dir):
    print("\n--- Analysis 2: Round-by-Round Success Rate ---")
    num_trials = len(all_logs)
    num_rounds = 0
    if all_logs:
        num_rounds = max(entry['round'] for logs in all_logs for entry in logs)

    # unambiguous_successes (unique top rate)
    unambiguous_successes = np.zeros((num_trials, num_rounds))
    for i, logs in enumerate(all_logs):
        for r in range(1, num_rounds + 1):
            round_selections = []
            for entry in logs:
                if entry["role"] == "Evaluator" and entry["round"] == r:
                    round_selections = parse_cards_from_msg(entry["msg"])
                    break
            if round_selections:
                counts = Counter(round_selections)
                top_two = counts.most_common(2)
                is_unambiguous = len(top_two) == 1 or (len(top_two) > 1 and top_two[0][1] > top_two[1][1])
                if is_unambiguous and top_two[0][0] == target_card_code:
                    unambiguous_successes[i, r-1] = 1

    empirical_rates = np.mean(unambiguous_successes, axis=0)

    print("\n--- LLM Simulation: Round-by-Round Success Rates ---")
    print(np.round(empirical_rates, 4))
    print("----------------------------------------------------")
    print("\n--- RMSE vs. Theoretical Models ---")
    rmse_poe = np.sqrt(np.mean((empirical_rates - theoretical_history_df['P(Target) - PoE'])**2))
    rmse_moe = np.sqrt(np.mean((empirical_rates - theoretical_history_df['P(Target) - MoE-EvalProd'])**2))
    rmse_mopoe = np.sqrt(np.mean((empirical_rates - theoretical_history_df['P(Target) - MoPoE'])**2))
    print(f"RMSE vs. PoE:       {rmse_poe:.4f}")
    print(f"RMSE vs. MoE:       {rmse_moe:.4f}")
    print(f"RMSE vs. MoPoE:     {rmse_mopoe:.4f}")
    print("-----------------------------------")

    # 95% CI
    std_error = np.std(unambiguous_successes, axis=0, ddof=1) / np.sqrt(num_trials)
    ci_95 = 1.96 * std_error
    rounds_axis = np.arange(1, num_rounds + 1)

    fig, ax = plt.subplots(figsize=(16, 6.2)) 
    _lk = dict(linewidth=2.2, markersize=7)

    ax.fill_between(
        rounds_axis, empirical_rates - ci_95, empirical_rates + ci_95,
        color='salmon', alpha=0.18, edgecolor='none',
        zorder=1
    )

    ax.plot(rounds_axis, empirical_rates, marker='o', linestyle='-',
            color='salmon', label='Empirical', zorder=3, clip_on=False, **_lk)
    ax.plot(rounds_axis, theoretical_history_df['P(Target) - PoE'], marker='s', linestyle='-',
            label='PoE', color='skyblue', zorder=5, clip_on=False, **_lk)
    ax.plot(rounds_axis, theoretical_history_df['P(Target) - MoE-EvalProd'], marker='^', linestyle='-',
            label='MoE', color='lightgreen', zorder=5, clip_on=False, **_lk)
    ax.plot(rounds_axis, theoretical_history_df['P(Target) - MoPoE'], marker='d', linestyle='-',
            label='MoPoE', color='plum', zorder=5, clip_on=False, **_lk)

    ax.set_xlabel('Round', fontsize=26)
    ax.set_ylabel(f'Probability of selecting {target_card_code}', fontsize=26)
    ax.set_xticks(rounds_axis)
    ax.set_xlim(1, num_rounds)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=22, loc='lower right')  

    fig.tight_layout()
    fig_path = out_dir / f"analysis_roundly_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved roundly success plot to: {fig_path}")
    plt.close(fig)

def analyze_and_plot_stepwise_success(all_logs, theoretical_history_df, target_card_code, out_dir):
    print("\n--- Analysis 2: Stepwise Success (proposal/selection) ---")
    num_trials = len(all_logs)
    num_rounds = 0
    if all_logs:
        num_rounds = max(entry.get('round', 0) for logs in all_logs for entry in logs)

    proposal = np.zeros((num_trials, num_rounds))   
    selection = np.zeros((num_trials, num_rounds))  

    for i, logs in enumerate(all_logs):
        for r in range(1, num_rounds + 1):
            proposed = []
            chosen = []
            for entry in logs:
                if entry.get("round") == r and entry.get("role") == "Sampler":
                    proposed = parse_cards_from_msg(entry.get("msg", ""))
                elif entry.get("round") == r and entry.get("role") == "Evaluator":
                    chosen = parse_cards_from_msg(entry.get("msg", ""))
            # proposal
            if proposed and target_card_code in proposed:
                proposal[i, r-1] = 1
            if chosen:
                counts = Counter(chosen)
                top_two = counts.most_common(2)
                is_unambiguous = len(top_two) == 1 or (len(top_two) > 1 and top_two[0][1] > top_two[1][1])
                if is_unambiguous and top_two[0][0] == target_card_code:
                    selection[i, r-1] = 1

    prop_rates = np.mean(proposal, axis=0)  
    sel_rates  = np.mean(selection, axis=0)  

    steps = np.arange(1, 2 * num_rounds + 1)
    interleaved_rates = np.empty(2 * num_rounds, dtype=float)
    interleaved_rates[0::2] = prop_rates  # odd steps: proposal
    interleaved_rates[1::2] = sel_rates   # even steps: selection

    if num_trials > 1:
        prop_se = np.std(proposal, axis=0, ddof=1) / np.sqrt(num_trials)
        sel_se  = np.std(selection, axis=0, ddof=1) / np.sqrt(num_trials)
    else:
        prop_se = np.zeros_like(prop_rates)
        sel_se  = np.zeros_like(sel_rates)
    interleaved_ci = np.empty_like(interleaved_rates)
    interleaved_ci[0::2] = 1.96 * prop_se
    interleaved_ci[1::2] = 1.96 * sel_se

    rmse_poe   = np.sqrt(np.mean((sel_rates - theoretical_history_df['P(Target) - PoE'])**2))
    rmse_moe   = np.sqrt(np.mean((sel_rates - theoretical_history_df['P(Target) - MoE-EvalProd'])**2))
    rmse_mopoe = np.sqrt(np.mean((sel_rates - theoretical_history_df['P(Target) - MoPoE'])**2))
    print("\n--- RMSE vs. Theoretical Models (selection steps) ---")
    print(f"RMSE vs. PoE:       {rmse_poe:.4f}")
    print(f"RMSE vs. MoE:       {rmse_moe:.4f}")
    print(f"RMSE vs. MoPoE:     {rmse_mopoe:.4f}")

    fig, ax = plt.subplots(figsize=(16, 6.2))
    _lk = dict(linewidth=2.2, markersize=7)

    # CI
    ax.fill_between(
        steps, interleaved_rates - interleaved_ci, interleaved_rates + interleaved_ci,
        color='salmon', alpha=0.18, edgecolor='none', label='95% CI (per step)', zorder=1
    )

    # empirical
    ax.plot(steps[0::2], interleaved_rates[0::2], marker='o', linestyle='-',
            color='gray', label='Sampler proposal (contains target)', zorder=3, **_lk)
    ax.plot(steps[1::2], interleaved_rates[1::2], marker='o', linestyle='-',
            color='salmon', label='Evaluator selection (unambiguous top=target)', zorder=3, **_lk)

    # theoretical
    step_even = steps[1::2]  
    ax.plot(step_even, theoretical_history_df['P(Target) - PoE'], marker='s', linestyle='--',
            label='Theoretical PoE', color='skyblue', zorder=2, **_lk)
    ax.plot(step_even, theoretical_history_df['P(Target) - MoE-EvalProd'], marker='^', linestyle=':',
            label='Theoretical MoE', color='lightgreen', zorder=2, **_lk)
    ax.plot(step_even, theoretical_history_df['P(Target) - MoPoE'], marker='d', linestyle='-.',
            label='Theoretical MoPoE', color='plum', zorder=2, **_lk)

    ax.set_xlabel('Step (odd: proposal, even: selection)', fontsize=22)
    ax.set_ylabel(f'P(target={target_card_code}) / success rate', fontsize=22)
    ax.set_title('Stepwise Trajectory: proposal → selection', fontsize=22)
    ax.set_xticks(steps)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=14, loc='lower right')

    fig.tight_layout()
    fig_path = out_dir / f"analysis_stepwise_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Saved stepwise success plot to: {fig_path}")
    plt.close(fig)

def analyze_and_plot_final_selection_histogram(
    all_logs,
    rounds,
    out_dir,
    highlight_card_code="D8",
    min_count=50,                 
    csv_out=True,
    target_color="salmon",        
    others_color="#C9C9C9",       
):

    # pool all selections
    all_final_selections = []
    for logs in all_logs:
        final_eval_selection = []
        for entry in logs:
            if entry.get("role") == "Evaluator" and entry.get("round") == rounds:
                final_eval_selection = parse_cards_from_msg(entry.get("msg", ""))
                break
        if final_eval_selection:
            all_final_selections.extend(final_eval_selection)

    total = len(all_final_selections)
    if total == 0:
        print("[WARN] No final-round selections found; skip histogram.")
        return None

    # count all
    from collections import Counter
    counts = Counter(all_final_selections)
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))  

    plot_items = [(c, n) for c, n in items if n >= min_count]
    if not plot_items:
        top_k = min(30, len(items))
        plot_items = items[:top_k]

    plot_cards = [c for c, _ in plot_items]
    plot_counts = [n for _, n in plot_items]

    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(len(plot_cards))
    colors = [target_color if c == highlight_card_code else others_color for c in plot_cards]

    fig_width = max(18, len(plot_cards) * 0.40)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    ax.bar(x, plot_counts, color=colors, edgecolor="none")

    ax.set_xlim(-0.5, len(x)-0.5)
    ax.set_xticks([]) 
    labels = [code_to_pretty(c) for c in plot_cards]
    if len(labels) <= 20: fs, rot = 20, 0
    elif len(labels) <= 35: fs, rot = 20, 0
    elif len(labels) <= 60: fs, rot = 13, 30
    else: fs, rot = 11, 45

    for xi, code, lab in zip(x, plot_cards, labels):
        kw = dict(ha='center', va='top', fontsize=fs, rotation=rot, annotation_clip=False)
        if code == highlight_card_code:
            kw.update(fontweight='bold', fontsize=22, color='salmon')  
        ax.annotate(lab, xy=(xi, 0), xytext=(0, -8), textcoords='offset points', **kw)

    ax.set_ylabel("Count", fontsize=27, labelpad=18)
    ax.tick_params(axis='y', labelsize=25)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    from matplotlib.ticker import MaxNLocator
    import matplotlib.ticker as mticker
    ax.set_ylim(0, max(plot_counts) * 1.03)  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 2.5, 5]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  

    from matplotlib.patches import Patch

    ax.set_ylim(top=max(plot_counts)*1.02) 
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = out_dir / f"analysis_final_selection_hist_v_{ts}.png"
    plt.savefig(png_path, dpi=200)
    print(f"Saved final selection histogram to: {png_path}")
    plt.close(fig)

    if csv_out:
        import csv
        csv_path = out_dir / f"final_selection_hist_{ts}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["card_code", "count", "probability", "plotted"])
            for c, n in items:
                w.writerow([c, n, f"{n/total:.6f}", "yes" if c in plot_cards else "no"])
        print(f"Saved histogram CSV to: {csv_path}")

    return png_path

def plot_card_heatmaps(all_logs, rounds, out_dir):
    """
    Spatial coverage of the card space (ranks × suits).
    Round 1 と Final の Sampler(提案) / Evaluator(選択) を 13×4 のヒートマップで並べて保存。
    保存先: out_dir / sir_heatmaps_YYYYmmdd_HHMMSS.png
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    def _dist_from_logs(role_key: str):
        from collections import Counter
        dists = []
        code_order = list(CARD_CODE_MAP.keys()) 
        for r in range(1, rounds + 1):
            cnt = Counter()
            total = 0
            for logs in all_logs:
                for e in logs:
                    if e.get("role") == role_key and e.get("round") == r:
                        cards = parse_cards_from_msg(e.get("msg", ""))
                        cnt.update(cards)
                        total += len(cards)
                        break
            if total == 0:
                p = np.ones(52, dtype=float) / 52.0
            else:
                p = np.array([cnt.get(code, 0) / total for code in code_order], dtype=float)
            dists.append(p)
        return np.array(dists)  

    suit_order = SUITS  
    rank_order = list(RANKS.keys())  
    suit_labels = ['♥', '♦', '♣', '♠']
    rank_short = { 'Jack':'J', 'Queen':'Q', 'King':'K', 'Ace':'A' }
    rank_labels = [rank_short.get(r, r) for r in rank_order]

    def _to_mat(p_vec):
        mat = np.zeros((4, 13), dtype=float)
        codes = list(CARD_CODE_MAP.keys())
        for i, code in enumerate(codes):
            name = CARD_CODE_MAP[code]       
            suit, rank = name.split()
            s = suit_order.index(suit)        
            r = rank_order.index(rank)        
            mat[s, r] = p_vec[i]
        return mat

    p_prop = _dist_from_logs("Sampler")     
    p_sel  = _dist_from_logs("Evaluator")   

    pairs = [
        ("Round 1 – Proposals", _to_mat(p_prop[0])),
        ("Round 1 – Selections", _to_mat(p_sel[0])),
        (f"Final (R{rounds}) – Proposals", _to_mat(p_prop[-1])),
        (f"Final (R{rounds}) – Selections", _to_mat(p_sel[-1])),
    ]

    vmax = max(m.max() for _, m in pairs)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for ax, (title, mat) in zip(axes, pairs):
        im = ax.imshow(mat, aspect='auto', cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(4));  ax.set_yticklabels(suit_labels)
        ax.set_xticks(range(13)); ax.set_xticklabels(rank_labels)
        ax.tick_params(axis='both', labelsize=10)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel('Probability', rotation=270, labelpad=12)

    out_path = out_dir / f"sir_heatmaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved SIR heatmaps to: {out_path}")

def plot_emp_theory_residual_final(all_logs, rounds, theory_final_vec, out_dir, theory_name="MoPoE"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from datetime import datetime

    # empirical final selection distribution ----
    def empirical_selection_dists(all_logs, rounds):
        from collections import Counter
        codes = list(CARD_CODE_MAP.keys())
        out = []
        for r in range(1, rounds+1):
            cnt = Counter(); total = 0
            for logs in all_logs:
                for e in logs:
                    if e.get("role") == "Evaluator" and e.get("round") == r:
                        cards = parse_cards_from_msg(e.get("msg",""))
                        cnt.update(cards); total += len(cards)
                        break
            if total == 0:
                p = np.ones(52)/52.0
            else:
                p = np.array([cnt.get(code,0)/total for code in codes], dtype=float)
            out.append(p)
        return np.array(out)

    suit_order = ['Heart','Diamond','Club','Spade']
    rank_order = ['2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace']
    suit_labels = ['♥','♦','♣','♠']
    rank_labels = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
    codes = list(CARD_CODE_MAP.keys())

    def to_mat(vec52):
        mat = np.zeros((4,13), dtype=float)
        for i, code in enumerate(codes):
            name = CARD_CODE_MAP[code]     
            suit, rank = name.split()
            s = suit_order.index(suit); r = rank_order.index(rank)
            mat[s, r] = vec52[i]
        return mat

    emp_final_52 = empirical_selection_dists(all_logs, rounds)[-1]
    th_final_52  = theory_final_vec
    emp_mat = to_mat(emp_final_52)
    th_mat  = to_mat(th_final_52)
    delta   = emp_mat - th_mat

    d8_row = suit_order.index('Diamond')
    d8_col = rank_order.index('8')

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)

    vmax = max(emp_mat.max(), th_mat.max())
    im0 = axes[0].imshow(emp_mat, aspect='auto', cmap='Reds', vmin=0, vmax=vmax)
    im1 = axes[1].imshow(th_mat,  aspect='auto', cmap='Reds', vmin=0, vmax=vmax)

    vmax_delta = np.abs(delta).max()
    im2 = axes[2].imshow(delta, aspect='auto', cmap='RdBu_r', vmin=-vmax_delta, vmax=vmax_delta)

    titles = [
        f"Empirical Selections (Final R{rounds})",
        f"Theory {theory_name} (Final)",
        "Residuals  (Emp − Theory)"
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(4));  ax.set_yticklabels(suit_labels, fontsize=10)
        ax.set_xticks(range(13)); ax.set_xticklabels(rank_labels, fontsize=10)
        ax.grid(False)

    for ax in axes:
        ax.add_patch(Rectangle((d8_col-0.5, d8_row-0.5), 1, 1, fill=False, ec='black', lw=1.5))
    axes[2].annotate('♦8', (d8_col, d8_row), xytext=(6, -8), textcoords='offset points',
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    cbar01 = fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
    cbar01.ax.set_ylabel('Probability', rotation=270, labelpad=12)
    cbar2  = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.set_ylabel('Δ Probability', rotation=270, labelpad=12)

    out = out_dir / f"emp_theory_residual_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved side-by-side comparison to: {out}")

def plot_emp_vs_theory_combined(theory_sim, all_logs, rounds, out_dir, theory_name="MoPoE"):
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Empirical: logs → (rounds, 52)
    def _emp_dist_from_logs(role_key: str):
        from collections import Counter
        code_order = list(CARD_CODE_MAP.keys())
        dists = []
        for r in range(1, rounds + 1):
            cnt = Counter()
            total = 0
            for logs in all_logs:
                for e in logs:
                    if e.get("role") == role_key and e.get("round") == r:
                        cards = parse_cards_from_msg(e.get("msg", ""))
                        cnt.update(cards)
                        total += len(cards)
                        break
            if total == 0:
                p = np.ones(52, dtype=float) / 52.0
            else:
                p = np.array([cnt.get(c, 0) / total for c in code_order], dtype=float)
            dists.append(p)
        return np.array(dists)  # (R,52)

    emp_prop = _emp_dist_from_logs("Sampler")
    emp_sel  = _emp_dist_from_logs("Evaluator")

    pS = theory_sim.p_sampler.copy()
    pE = theory_sim.p_evaluator.copy()

    b_poe = np.ones(52) / 52.0
    b_moe = np.ones(52) / 52.0

    sel_R1 = None
    sel_final = None

    for t in range(1, rounds + 1):
        b_poe = b_poe * (pS * pE)
        b_poe /= b_poe.sum()

        mixed = theory_sim.beta * pS + (1 - theory_sim.beta) * b_moe
        b_moe = pE * mixed
        b_moe /= b_moe.sum()

        # MoPoE
        b_mopoe = theory_sim.alpha * b_poe + (1 - theory_sim.alpha) * b_moe

        if t == 1:
            sel_R1 = dict(
                PoE=b_poe.copy(),
                MoE=b_moe.copy(),
                MoPoE=b_mopoe.copy()
            )
        if t == rounds:
            sel_final = dict(
                PoE=b_poe.copy(),
                MoE=b_moe.copy(),
                MoPoE=b_mopoe.copy()
            )

    th_sel_R1    = sel_R1.get(theory_name, sel_R1["MoPoE"])
    th_sel_final = sel_final.get(theory_name, sel_final["MoPoE"])
    th_prop_R1 = pS
    th_prop_final = pS

    suit_order = ['Heart','Diamond','Club','Spade']
    rank_order = ['2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace']
    rank_short = { 'Jack':'J', 'Queen':'Q', 'King':'K', 'Ace':'A' }
    rank_labels = [rank_short.get(r, r) for r in rank_order]
    suit_labels = ['♥','♦','♣','♠']
    codes = list(CARD_CODE_MAP.keys())

    def _vec52_to_mat(p_vec):
        mat = np.zeros((4, 13), dtype=float)
        for i, code in enumerate(codes):
            name = CARD_CODE_MAP[code]
            suit, rank = name.split()
            s = suit_order.index(suit)
            r = rank_order.index(rank)
            mat[s, r] = p_vec[i]
        return mat

    def _concat_prop_sel(prop_vec, sel_vec):
        L = _vec52_to_mat(prop_vec)
        R = _vec52_to_mat(sel_vec)
        gap = np.zeros((4, 1))  
        return np.hstack([L, gap, R]) 

    emp_R1_mat    = _concat_prop_sel(emp_prop[0], emp_sel[0])
    emp_final_mat = _concat_prop_sel(emp_prop[-1], emp_sel[-1])
    th_R1_mat     = _concat_prop_sel(th_prop_R1,  th_sel_R1)
    th_final_mat  = _concat_prop_sel(th_prop_final, th_sel_final)

    import numpy as np
    from matplotlib.colors import PowerNorm

    stack = np.hstack([
        emp_R1_mat.ravel(), emp_final_mat.ravel(),
        th_R1_mat.ravel(),  th_final_mat.ravel()
    ])

    vmax = np.quantile(stack, 0.995)
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.2), constrained_layout=True)

    panels = [
        (axes[0,0], emp_R1_mat,    f"Empirical  — Round 1"),
        (axes[1,0], emp_final_mat, f"Empirical  — Final (R{rounds})"),
        (axes[0,1], th_R1_mat,     f"Theory {theory_name} — Round 1"),
        (axes[1,1], th_final_mat,  f"Theory {theory_name} — Final (R{rounds})"),
    ]

    for ax, mat, title in panels:
        im = ax.imshow(mat, aspect='auto', cmap='viridis', norm=norm)
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(4));  ax.set_yticklabels(suit_labels, fontsize=10)
        xticks = list(range(0, 13, 2)) + list(range(14, 27, 2))
        xtlbls = [rank_labels[i] for i in range(0,13,2)] + [rank_labels[i-14] for i in range(14,27,2)]
        ax.set_xticks(xticks); ax.set_xticklabels(xtlbls, fontsize=10)
        ax.axvline(13+0.5, color='k', lw=1, alpha=0.25)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Probability', rotation=270, labelpad=12)

    out = out_dir / f"emp_vs_theory_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved Empirical vs Theory combined heatmaps to: {out}")

def plot_empirical_coverage_only(all_logs, rounds, out_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from datetime import datetime

    def _dist_from_logs(role_key: str):
        from collections import Counter
        code_order = list(CARD_CODE_MAP.keys())
        dists = []
        for r in range(1, rounds + 1):
            cnt = Counter(); total = 0
            for logs in all_logs:
                for e in logs:
                    if e.get("role") == role_key and e.get("round") == r:
                        cards = parse_cards_from_msg(e.get("msg", ""))
                        cnt.update(cards); total += len(cards)
                        break
            if total == 0:
                p = np.ones(52, dtype=float) / 52.0
            else:
                p = np.array([cnt.get(c, 0) / total for c in code_order], dtype=float)
            dists.append(p)
        return np.array(dists)  

    emp_prop = _dist_from_logs("Sampler")
    emp_sel  = _dist_from_logs("Evaluator")

    suit_order = ['Heart','Diamond','Club','Spade']
    rank_order = ['2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace']
    rank_short = { 'Jack':'J', 'Queen':'Q', 'King':'K', 'Ace':'A' }
    rank_labels = [rank_short.get(r, r) for r in rank_order]
    suit_labels = ['♥','♦','♣','♠']
    codes = list(CARD_CODE_MAP.keys())

    def _vec52_to_mat(p_vec):
        mat = np.zeros((4, 13), dtype=float)
        for i, code in enumerate(codes):
            name = CARD_CODE_MAP[code]
            suit, rank = name.split()
            s = suit_order.index(suit)
            r = rank_order.index(rank)
            mat[s, r] = p_vec[i]
        return mat

    def _concat_prop_sel(prop_vec, sel_vec):
        L = _vec52_to_mat(prop_vec)
        R = _vec52_to_mat(sel_vec)
        gap = np.zeros((4, 1))  
        return np.hstack([L, gap, R])  

    mat_R1    = _concat_prop_sel(emp_prop[0],  emp_sel[0])
    mat_final = _concat_prop_sel(emp_prop[-1], emp_sel[-1])

    stack = np.hstack([mat_R1.ravel(), mat_final.ravel()])
    vmax = np.quantile(stack, 0.995)
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), constrained_layout=True)

    panels = [
        (axes[0], mat_R1,    "Empirical  —  Round 1"),
        (axes[1], mat_final, f"Empirical  —  Final (R{rounds})"),
    ]

    from matplotlib.colors import LinearSegmentedColormap
    soft_purple = LinearSegmentedColormap.from_list(
        "soft_purple",
        ["#FCFBFF", "#EDE8F6", "#D7CEF0", "#B8A9E0", "#957ACC", "#6E58B7", "#5B46A7"]
    )

    for ax, mat, title in panels:
        im = ax.imshow(mat, aspect='auto', cmap=soft_purple, norm=norm)
        ax.set_title(title, fontsize=14)
        ax.set_yticks(range(4));  ax.set_yticklabels(suit_labels, fontsize=11)
        xticks = list(range(0, 13, 2)) + list(range(14, 27, 2))
        xtlbls = [rank_labels[i] for i in range(0,13,2)] + [rank_labels[i-14] for i in range(14,27,2)]
        ax.set_xticks(xticks); ax.set_xticklabels(xtlbls, fontsize=11)
        ax.axvline(13+0.5, color='k', lw=1, alpha=0.25)
        ax.text(6.5, -0.7, "Proposals", ha='center', va='center', fontsize=15)
        ax.text(13+1+6.5, -0.7, "Selections", ha='center', va='center', fontsize=15)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Probability', rotation=270, labelpad=12)

    out = out_dir / f"empirical_coverage_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved empirical-only coverage to: {out}")

def export_micro_heatmaps_round1_square(
    all_logs, rounds, out_dir,
    sampler_cmap='Blues', evaluator_cmap='Reds',
    size_inch=1.8, dpi=300,
    gamma=0.60, tick_every=2,
    show_d8_box=False
):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from datetime import datetime
    from matplotlib.patches import Rectangle

    def _dist_from_logs(role_key: str):
        from collections import Counter
        code_order = list(CARD_CODE_MAP.keys())
        dists = []
        for r in range(1, rounds + 1):
            cnt = Counter(); total = 0
            for logs in all_logs:
                for e in logs:
                    if e.get("role") == role_key and e.get("round") == r:
                        cards = parse_cards_from_msg(e.get("msg", ""))
                        cnt.update(cards); total += len(cards)
                        break
            if total == 0:
                p = np.ones(52, dtype=float) / 52.0
            else:
                p = np.array([cnt.get(c, 0) / total for c in code_order], dtype=float)
            dists.append(p)
        return np.array(dists)

    sampler = _dist_from_logs("Sampler")[0]
    evaluator = _dist_from_logs("Evaluator")[0]

    suit_order = ['Heart','Diamond','Club','Spade']
    rank_order = ['2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace']
    suit_labels = ['♥','♦','♣','♠']
    rank_short = {'Jack':'J','Queen':'Q','King':'K','Ace':'A'}
    rank_labels = [rank_short.get(r, r) for r in rank_order]
    codes = list(CARD_CODE_MAP.keys())

    def _vec52_to_mat(p_vec):
        mat = np.zeros((4, 13), dtype=float)
        for i, code in enumerate(codes):
            name = CARD_CODE_MAP[code]
            suit, rank = name.split()
            s = suit_order.index(suit); r = rank_order.index(rank)
            mat[s, r] = p_vec[i]
        return mat

    mat_S = _vec52_to_mat(sampler)
    mat_E = _vec52_to_mat(evaluator)

    stack = np.hstack([mat_S.ravel(), mat_E.ravel()])
    vmax  = np.quantile(stack, 0.995)
    norm  = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    def _save_square(mat, cmap, fname, y_on_right=False):
        fig = plt.figure(figsize=(size_inch, size_inch), dpi=dpi, facecolor=(1,1,1,0))
        if y_on_right:
            ax = fig.add_axes([0.10, 0.20, 0.72, 0.72])  
        else:
            ax = fig.add_axes([0.22, 0.20, 0.72, 0.72])  

        im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm)

        if y_on_right:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
        ax.set_yticks(range(4)); ax.set_yticklabels(suit_labels, fontsize=15)

        from matplotlib.ticker import FixedLocator, NullFormatter, MultipleLocator

        ax.xaxis.set_major_locator(FixedLocator([0, 12]))
        ax.set_xticklabels([rank_labels[0], rank_labels[12]], fontsize=15)

        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='x', which='minor', length=2, width=0.5, color=(0,0,0,0.5))

        ax.tick_params(length=2.5, width=0.6)

        ax.xaxis.set_minor_locator(MultipleLocator(tick_every))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='x', which='minor', length=2, width=0.5,
                       color=(0,0,0,0.5))

        for side, spine in ax.spines.items():
            spine.set_linewidth(0.6); spine.set_alpha(0.4)

        if show_d8_box:
            d8_row = suit_order.index('Diamond'); d8_col = rank_order.index('8')
            ax.add_patch(Rectangle((d8_col-0.5, d8_row-0.5), 1, 1,
                                   fill=False, ec='white', lw=0.7, alpha=0.7))

        out = out_dir / fname
        fig.savefig(out, dpi=dpi, transparent=True)  
        plt.close(fig)
        print(f"Saved: {out}")

    _save_square(mat_S, sampler_cmap,   "sampler_microheatmap_R1_sq.png",   y_on_right=True)
    _save_square(mat_E, evaluator_cmap, "evaluator_microheatmap_R1_sq.png", y_on_right=False)

def main():
    parser = argparse.ArgumentParser(description="Analyze card game logs and compare with theoretical models.")

    parser.add_argument(
        "--logs",
        required=True,
        nargs="+",
        help="One or more directories (or glob patterns) containing trial_*.json."
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds the simulation was run for.")
    parser.add_argument("--tau_e", type=float, default=2.5)
    parser.add_argument("--tau_s_ratio", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for MoPoE model.")
    parser.add_argument("--target_card_code", type=str, default="D8")
    args = parser.parse_args()

    def _collect_log_paths(paths):
        collected = []
        for p in paths:
            if os.path.isdir(p):
                collected += glob.glob(os.path.join(p, "trial_*.json"))
            else:
                collected += glob.glob(p)
        collected = sorted(set(collected))
        return collected

    log_paths = _collect_log_paths(args.logs)
    if not log_paths:
        raise SystemExit(f"Error: No log files found under: {args.logs}")

    folders = sorted(set(os.path.dirname(p) for p in log_paths))
    print(f"Found {len(log_paths)} log files across {len(folders)} folder(s).")
    for f in folders:
        cnt = sum(1 for p in log_paths if os.path.dirname(p) == f)
        print(f"  - {f}: {cnt} files")

    common_parent = os.path.commonpath([os.path.abspath(os.path.dirname(p)) for p in log_paths])
    out_dir = Path(common_parent) / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved under: {out_dir}")

    target_card_name = CARD_CODE_MAP.get(args.target_card_code)
    if not target_card_name:
        raise SystemExit(f"Error: Target card code '{args.target_card_code}' is invalid.")

    theory_sim = CardExperimentSimulator(
        temp_sampler=(args.tau_e * args.tau_s_ratio),
        temp_evaluator=args.tau_e,
        beta=args.beta,
        alpha=args.alpha
    )
    theoretical_history_df, theoretical_final_dist_df = theory_sim.run_simulation(
        target_card_name=target_card_name, rounds=args.rounds
    )

    all_logs = [json.load(open(p, 'r', encoding='utf-8'))['events'] for p in log_paths]
    num_trials = len(all_logs) 

    import numpy as np
    codes = list(CARD_CODE_MAP.keys())
    name_to_idx = { CARD_CODE_MAP[c]: i for i, c in enumerate(codes) }

    theory_col = 'Final_P(MoPoE)'   
    theory_final_vec = np.zeros(52, dtype=float)
    for _, row in theoretical_final_dist_df.iterrows():
        idx = name_to_idx[row['Card']]     
        theory_final_vec[idx] = float(row[theory_col])

    plot_emp_theory_residual_final(
        all_logs, args.rounds,
        theory_final_vec=theory_final_vec,  
        out_dir=out_dir,
        theory_name="MoPoE"
    )
    print(f"\n--- [Reference] Theoretical Probabilities (alpha={args.alpha}) over {args.rounds} Rounds ---")
    print(theoretical_history_df.to_string(index=False))
    
    print(f"\n--- [Reference] Theoretical Final Distribution (alpha={args.alpha}) after {args.rounds} Rounds (Top 10) ---")
    print(theoretical_final_dist_df.head(10).to_string(index=False))

    unambiguous_success_count = 0 

    all_final_winners = []
    for logs in all_logs:
        final_eval_selection = []
        for entry in reversed(logs):
            if entry.get("role") == "Evaluator" and entry.get("round") == args.rounds:
                final_eval_selection = parse_cards_from_msg(entry.get("msg", ""))
                break
        
        if final_eval_selection:
            counts = Counter(final_eval_selection)
            if not counts:
                continue
            top_two = counts.most_common(2)
            is_unambiguous_win = False
            if len(top_two) == 1: 
                is_unambiguous_win = True
            elif top_two[0][1] > top_two[1][1]: 
                is_unambiguous_win = True
            
            if is_unambiguous_win and top_two[0][0] == args.target_card_code:
                unambiguous_success_count += 1

            max_count = counts.most_common(1)[0][1]
            
            top_tied_cards = [card for card, count in counts.items() if count == max_count]
            
            if args.target_card_code in top_tied_cards:
                winner = args.target_card_code
            else:
                winner = top_tied_cards[0]
            
            all_final_winners.append(winner)
    print("\n--- Summary of Trial Winners ---")
    winner_counts = Counter(all_final_winners)
    for card, count in winner_counts.most_common():
        print(f"{card}: {count} times")
    print("---------------------------------")

    print("\n--- Unambiguous Success Rate (Final Round) ---")
    unambiguous_rate = (unambiguous_success_count / num_trials) * 100
    print(f"Rate at which D8 was the SOLE top choice: {unambiguous_rate:.2f}% ({unambiguous_success_count}/{num_trials})")
    print("-------------------------------------------------")
            
    analyze_and_plot_roundly_success(all_logs, theoretical_history_df, args.target_card_code, out_dir)

    analyze_and_plot_final_distribution(all_final_winners, theoretical_final_dist_df, out_dir)
    analyze_and_plot_stepwise_success(all_logs, theoretical_history_df, args.target_card_code, out_dir)

    analyze_and_plot_final_selection_histogram(
        all_logs, args.rounds, out_dir,
        highlight_card_code="D8",
        min_count=50,        
        csv_out=True
    )

    plot_card_heatmaps(all_logs, args.rounds, out_dir)

    plot_emp_vs_theory_combined(
        theory_sim=theory_sim,
        all_logs=all_logs,
        rounds=args.rounds,
        out_dir=out_dir,
        theory_name="MoPoE"   
    )

    plot_empirical_coverage_only(all_logs, args.rounds, out_dir)

    export_micro_heatmaps_round1_square(all_logs, args.rounds, out_dir,
                                sampler_cmap='Blues', evaluator_cmap='Reds')


    print("\n--- Analysis Complete ---")
    
if __name__ == "__main__":
    main()