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
import matplotlib as mpl
from collections import Counter
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe

# Config
FIXED_VMAX_OVERRIDE = 0.3540

SIZE_SCALE           = 20000          
PRIOR_SIZE_SCALE     = 20000          
PRIOR_MIN_PROB       = 0.01           

GRID_RIGHT_MARGIN    = 0.93           
AX_TICKS = [0, 5, 10]

# font size
AXIS_LABEL_FONTSIZE       = 30
TICK_LABEL_FONTSIZE       = 30
PANEL_TITLE_FONTSIZE      = 30
COLORBAR_LABEL_FONTSIZE   = 22
COLORBAR_TICK_FONTSIZE    = 26

LABEL_FONT_SIZE           = 28   
LABEL_TOPK                = 2   
LABEL_MIN_WEIGHT          = 0.0  
LABEL_DX, LABEL_DY        = 0.10, 0.10  

# ---- Colors / Markers ----
PRIOR_SAMPLER_FACE_HEX   = "#1f77b4"
PRIOR_SAMPLER_FACE_ALPHA = 0.25
PRIOR_SAMPLER_EDGE_HEX   = "#ffffff"
PRIOR_SAMPLER_EDGE_LW    = 0.4
PRIOR_SAMPLER_MARKER     = "o"        # ■

PRIOR_EVAL_FACE_HEX      = "#d62728"
PRIOR_EVAL_FACE_ALPHA    = 0.25
PRIOR_EVAL_EDGE_HEX      = "#ffffff"
PRIOR_EVAL_EDGE_LW       = 0.4
PRIOR_EVAL_MARKER        = "o"        # ●

EVALUATOR_CMAP_ROUND  = "viridis"
EVALUATOR_ALPHA_ROUND = 0.85

# Country utilities
COUNTRY_LIST = [
    "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua and Barbuda","Argentina","Armenia","Australia",
    "Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bhutan",
    "Bolivia","Bosnia and Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina Faso","Burundi","Cabo Verde",
    "Cambodia","Cameroon","Canada","Central African Republic","Chad","Chile","China","Colombia","Comoros",
    "Congo, Democratic Republic of the","Congo, Republic of the","Costa Rica","Croatia","Cuba","Cyprus","Czech Republic",
    "Denmark","Djibouti","Dominica","Dominican Republic","Ecuador","Egypt","El Salvador","Equatorial Guinea","Eritrea",
    "Estonia","Eswatini","Ethiopia","Fiji","Finland","France","Gabon","Gambia","Georgia","Germany","Ghana","Greece",
    "Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti","Honduras","Hungary","Iceland","India","Indonesia",
    "Iran","Iraq","Ireland","Israel","Italy","Ivory Coast","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kiribati",
    "Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania",
    "Luxembourg","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands","Mauritania","Mauritius",
    "Mexico","Micronesia","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nauru",
    "Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Korea","North Macedonia","Norway","Oman",
    "Pakistan","Palau","Palestine","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal",
    "Qatar","Romania","Russia","Rwanda","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa",
    "San Marino","Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore",
    "Slovakia","Slovenia","Solomon Islands","Somalia","South Africa","South Korea","South Sudan","Spain","Sri Lanka",
    "Sudan","Suriname","Sweden","Switzerland","Syria","Taiwan","Tajikistan","Tanzania","Thailand","Timor-Leste","Togo",
    "Tonga","Trinidad and Tobago","Tunisia","Turkey","Turkmenistan","Tuvalu","Uganda","Ukraine","United Arab Emirates",
    "United Kingdom","United States","Uruguay","Uzbekistan","Vanuatu","Vatican City","Venezuela","Vietnam","Yemen",
    "Zambia","Zimbabwe"
]
COUNTRY_INDEX = {n:i for i,n in enumerate(COUNTRY_LIST)}
COUNTRY_PAT   = re.compile(r"Country List:\s*\[([^\]]+)\]", re.I)

def parse_countries_from_msg(text: str) -> list[str]:
    m = COUNTRY_PAT.search(text or "")
    if not m:
        return []
    countries = [item.strip().strip("'\"") for item in m.group(1).split(",")]
    return [c for c in countries if c in COUNTRY_INDEX]

# Prob helpers
def _ensure_prob(x, eps=1e-12):
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    s = x.sum()
    return x / (s if s > 0 else eps)

def boltzmann_selection(utilities, temperature=1.0):
    utilities = np.array(utilities, dtype=float)
    if temperature <= 0:
        temperature = 1e-8
    z = utilities / temperature
    z -= np.max(z)  
    p = np.exp(z)
    return _ensure_prob(p)

def _collect_probs(all_events_data, round_num, role):
    sel = [c for events in all_events_data
           for e in events
           if e.get('round') == round_num and e.get('role') == role
           for c in parse_countries_from_msg(e.get("msg", ""))]
    if not sel:
        return {}
    tot = len(sel)
    return {c: cnt / tot for c, cnt in Counter(sel).items()}

_EXCLUDE_FOR_XY = {'trial','country','score','evaluator_init_prob','sampler_init_prob','evaluator_prob'}
def _pick_xy_cols(df: pd.DataFrame):
    cols = list(df.columns)
    numeric_candidates = []
    for c in cols:
        if c in _EXCLUDE_FOR_XY: continue
        ser = pd.to_numeric(df[c], errors='coerce')
        if ser.notna().mean() >= 0.9:
            numeric_candidates.append(c)
    if len(numeric_candidates) < 2:
        raise SystemExit("you need at least two columns for scores_csv")
    x_col, y_col = numeric_candidates[0], numeric_candidates[1]
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    return x_col, y_col

def _weighted_centroid(df, prob_col, x_col, y_col):
    w = df[prob_col].to_numpy(float)
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)
    s = w.sum()
    if s <= 0: return None
    return float((w*x).sum()/s), float((w*y).sum()/s)

 
def _abbr_country(name: str) -> str:
    if not name:
        return ""
    parts = [p for p in name.replace(',', '').split() if p]
    if len(parts) >= 2:
        return ''.join(p[0] for p in parts).upper()
    w = parts[0]
    return (w[:3]).upper()

def _label_topk(ax, df, x_col, y_col, weight_col, topk=LABEL_TOPK):
    if weight_col not in df.columns:
        return
    tmp = df[[x_col, y_col, 'country', weight_col]].copy()
    tmp = tmp[tmp[weight_col] > LABEL_MIN_WEIGHT]
    if tmp.empty:
        return
    tmp = tmp.sort_values(by=weight_col, ascending=False).head(topk)
    bboxes = []
    for _, r in tmp.iterrows():
        x, y = float(r[x_col]), float(r[y_col])
        text = _abbr_country(str(r['country']))
        t = ax.annotate(
            text, (x, y),
            textcoords="offset points", xytext=(6, 6), xycoords="data",
            fontsize=LABEL_FONT_SIZE, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.2', linewidth=0),
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
        )
        max_iter = 60
        candidates = [(6,6),(10,2),(2,10),(-6,6),(-10,2),(6,-6),(-6,-6),(10,10)]
        for _ in range(max_iter):
            fig = ax.figure
            renderer = fig.canvas.get_renderer()
            bb = t.get_window_extent(renderer=renderer)
            hit = False
            for prev in bboxes:
                if bb.overlaps(prev):
                    hit = True
                    break
            if not hit:
                bboxes.append(bb)
                break
            xoff, yoff = candidates[0]
            candidates = candidates[1:] + [ (xoff+2, yoff+2) ]
            t.set_position((xoff, yoff))  

    ax.figure.canvas.draw_idle()

def _style_axes(ax, x_col, y_col):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([0,5,10]); ax.set_yticks([0,5,10])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which='major', ls='--', alpha=0.35)
    ax.grid(True, which='minor', ls=':',  alpha=0.15)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

    try:
        ax.set_box_aspect(1)          
    except Exception:
        ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(x_col.replace('_',' ').title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_col.replace('_',' ').title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.xaxis.labelpad = 6; ax.yaxis.labelpad = 6



def _scatter_panel(ax, df, x_col, y_col, norm, vmax,
                   mode='round', add_legend=False, show_centroids=False):
    if mode == 'prior':
        df = df.copy()
        if 'sampler_init_prob' in df.columns:
            df.loc[df['sampler_init_prob'] < PRIOR_MIN_PROB, 'sampler_init_prob'] = 0.0
        if 'evaluator_init_prob' in df.columns:
            df.loc[df['evaluator_init_prob'] < PRIOR_MIN_PROB, 'evaluator_init_prob'] = 0.0

        eps = max(vmax, 1e-12)

        # Sampler（■）
        if 'sampler_init_prob' in df.columns:
            ssz = (df['sampler_init_prob'] * PRIOR_SIZE_SCALE / eps)
            face_rgba = mpl.colors.to_rgba(PRIOR_SAMPLER_FACE_HEX, PRIOR_SAMPLER_FACE_ALPHA)
            edge_rgba = mpl.colors.to_rgba(PRIOR_SAMPLER_EDGE_HEX, 1.0)
            ax.scatter(
                df[x_col], df[y_col],
                s=ssz, marker=PRIOR_SAMPLER_MARKER,
                facecolors=[face_rgba]*len(df),
                edgecolors=[edge_rgba]*len(df),
                linewidths=PRIOR_SAMPLER_EDGE_LW, zorder=1
            )

        # Evaluator（●）
        if 'evaluator_init_prob' in df.columns:
            esz = (df['evaluator_init_prob'] * PRIOR_SIZE_SCALE / eps)
            face_rgba = mpl.colors.to_rgba(PRIOR_EVAL_FACE_HEX, PRIOR_EVAL_FACE_ALPHA)
            edge_rgba = mpl.colors.to_rgba(PRIOR_EVAL_EDGE_HEX, 1.0)
            ax.scatter(
                df[x_col], df[y_col],
                s=esz, marker=PRIOR_EVAL_MARKER,
                facecolors=[face_rgba]*len(df),
                edgecolors=[edge_rgba]*len(df),
                linewidths=PRIOR_EVAL_EDGE_LW, zorder=2
            )

        if add_legend:
            handles = [
                Line2D([0],[0], marker=PRIOR_SAMPLER_MARKER, linestyle='',
                       markerfacecolor=PRIOR_SAMPLER_FACE_HEX, markeredgecolor='white',
                       markeredgewidth=PRIOR_SAMPLER_EDGE_LW, markersize=10, label='Sampler Prior'),
                Line2D([0],[0], marker=PRIOR_EVAL_MARKER, linestyle='',
                       markerfacecolor=PRIOR_EVAL_FACE_HEX, markeredgecolor='white',
                       markeredgewidth=PRIOR_EVAL_EDGE_LW, markersize=10, label='Evaluator Prior'),
            ]
            ax.legend(handles=handles, fontsize=12, loc='upper left',
                      bbox_to_anchor=(0.03, 0.97), borderaxespad=0., frameon=True)

    else: 
        eps = max(vmax, 1e-12)
        val = df['evaluator_prob'] if 'evaluator_prob' in df.columns else pd.Series(0, index=df.index)
        cmap = mpl.colormaps[EVALUATOR_CMAP_ROUND]
        ax.scatter(
            df[x_col], df[y_col],
            s=(val * SIZE_SCALE / eps),
            c=val, cmap=cmap, norm=norm,
            alpha=EVALUATOR_ALPHA_ROUND, edgecolors='w', linewidths=0.4, zorder=2
        )
        if show_centroids and 'evaluator_prob' in df.columns:
            e_cent = _weighted_centroid(df, 'evaluator_prob', x_col, y_col)
            if e_cent:
                ax.scatter([e_cent[0]], [e_cent[1]], s=220, marker='*',
                           edgecolors='k', linewidths=0.8, facecolors='gold', zorder=4)

        _label_topk(ax, df, x_col, y_col, 'evaluator_prob', topk=LABEL_TOPK)
 
def _load_eval_init_series(csv_path: str, tau_e: float) -> pd.Series:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if 'evaluator_init_prob' in df.columns:
        cname = 'country' if 'country' in df.columns else df.columns[0]
        s = df.set_index(cname)['evaluator_init_prob'].reindex(COUNTRY_LIST).fillna(0.0).astype(float)
        return pd.Series(_ensure_prob(s.to_numpy()), index=COUNTRY_LIST, name='evaluator_init_prob')

    if 'country' in df.columns and 'score' in df.columns:
        g = df.groupby('country')['score'].mean()
        utilities = np.array([g.get(c, 0.0) for c in COUNTRY_LIST], dtype=float)
        p = boltzmann_selection(utilities, temperature=tau_e)
        return pd.Series(p, index=COUNTRY_LIST, name='evaluator_init_prob')

    print(f"[WARN] {csv_path}: evaluator_init_prob も trial/country/score も見つからず。Prior Evaluator は表示されません。")
    return pd.Series(np.zeros(len(COUNTRY_LIST)), index=COUNTRY_LIST, name='evaluator_init_prob')

# 3x2 grid
def generate_combined_grid6(
    df_scores_unb, events_unb, df_scores_bia, events_bia,
    eval_init_unb: pd.Series, eval_init_bia: pd.Series,
    outdir, scenario_id=None, vmin=0.0, vmax=0.35,
    sampler_init_round=1
):
    fig = plt.figure(figsize=(13.0, 19))  
    gs  = fig.add_gridspec(
        nrows=3, ncols=3,
        left=0.075, right=0.915, bottom=0.145, top=0.965,
        width_ratios=[1, 0.012, 1],  
        wspace=0.00, hspace=0.10
    )
    axs = np.empty((3, 2), dtype=object)
    for r in range(3):
        axs[r, 0] = fig.add_subplot(gs[r, 0])  
        axs[r, 1] = fig.add_subplot(gs[r, 2])  

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    x_unb, y_unb = _pick_xy_cols(df_scores_unb)
    x_bia, y_bia = _pick_xy_cols(df_scores_bia)

    def _build_prior_df(df_scores, events, eval_init_s):
        df = df_scores.copy()
        sp_init = _collect_probs(events, sampler_init_round, 'Sampler')
        sampler_init_s = pd.Series(sp_init, name='sampler_init_prob')
        df['sampler_init_prob']   = df['country'].map(sampler_init_s).fillna(0.0)
        df['evaluator_init_prob'] = df['country'].map(eval_init_s).fillna(0.0)
        return df

    def _build_round_df(df_scores, events, round_num):
        df = df_scores.copy()
        ep = _collect_probs(events, round_num, 'Evaluator')
        df['evaluator_prob'] = df['country'].map(ep).fillna(0.0)
        return df

    df_unb_prior = _build_prior_df(df_scores_unb, events_unb, eval_init_unb)
    df_unb_r1    = _build_round_df(df_scores_unb, events_unb, 1)
    df_unb_r10   = _build_round_df(df_scores_unb, events_unb, 10)
    df_bia_prior = _build_prior_df(df_scores_bia, events_bia, eval_init_bia)
    df_bia_r1    = _build_round_df(df_scores_bia, events_bia, 1)
    df_bia_r10   = _build_round_df(df_scores_bia, events_bia, 10)

    panels = [
        ('Prior',    df_unb_prior, x_unb, y_unb, 'prior', (0,0), False,  False),
        ('Prior',    df_bia_prior, x_bia, y_bia, 'prior', (0,1), False,  False),
        ('Round 1',  df_unb_r1,    x_unb, y_unb, 'round', (1,0), False,  True),
        ('Round 1',  df_bia_r1,    x_bia, y_bia, 'round', (1,1), False,  True),
        ('Round 10', df_unb_r10,   x_unb, y_unb, 'round', (2,0), False,  True),
        ('Round 10', df_bia_r10,   x_bia, y_bia, 'round', (2,1), False,  True),
    ]

    legend_artists = []
    for title, dfp, x_col, y_col, mode, (ri, ci), add_leg, show_cent in panels:
        ax = axs[ri, ci]
        _scatter_panel(ax, dfp, x_col, y_col, norm, vmax,
                       mode=mode, add_legend=add_leg, show_centroids=show_cent)
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, pad=6)
        _style_axes(ax, x_col, y_col)

        if ri < 2:
            ax.set_xlabel(""); ax.tick_params(axis='x', labelbottom=False)
        if ci == 1:
            ax.set_ylabel(""); ax.tick_params(axis='y', labelleft=False)

        if title == 'Prior' and ri == 0:
            if ci == 0:
                sampler_label   = r'$P_S(x \mid \theta_S^{\mathrm{Unbiased}})$'
                evaluator_label = r'$P_E(x \mid \theta_E)$'
                loc, bta = ('center right', (-0.35, 0.5))
            else:
                sampler_label   = r'$P_S(x \mid \theta_S^{\mathrm{Biased}})$'
                evaluator_label = r'$P_E(x \mid \theta_E)$'
                loc, bta = ('center left', (1.15, 0.5))

            handles = [
                Line2D([0],[0], marker=PRIOR_SAMPLER_MARKER, linestyle='',
                       markerfacecolor=PRIOR_SAMPLER_FACE_HEX, markeredgecolor='white',
                       markeredgewidth=PRIOR_SAMPLER_EDGE_LW, markersize=10, label=sampler_label),
                Line2D([0],[0], marker=PRIOR_EVAL_MARKER, linestyle='',
                       markerfacecolor=PRIOR_EVAL_FACE_HEX, markeredgecolor='white',
                       markeredgewidth=PRIOR_EVAL_EDGE_LW, markersize=10, label=evaluator_label),
            ]
            leg = ax.legend(handles=handles, fontsize=24, loc=loc,
                            bbox_to_anchor=bta, borderaxespad=0., frameon=True)
            leg.set_in_layout(False)
            legend_artists.append(leg)

    plt.draw()
    mid_bbox = fig.add_subplot(gs[:, 1]).get_position()
    plt.delaxes(fig.axes[-1])
    x_sep = (mid_bbox.x0 + mid_bbox.x1) / 2.0
    y0 = min(axs[2,0].get_position().y0, axs[2,1].get_position().y0)
    y1 = max(axs[0,0].get_position().y1, axs[0,1].get_position().y1)
    sep_line = Line2D([x_sep, x_sep], [y0, y1], transform=fig.transFigure,
                      linestyle='-', linewidth=1.0, color='#BBBBBB')
    fig.add_artist(sep_line)

    fig.text(0.25, 0.045,
             r'$P^{(t)} \propto P_S(x \mid \theta_S^{\mathrm{Unbiased}}, c)\, P_E(x \mid \theta_E)$',
             ha='center', va='center', fontsize=24)
    fig.text(0.75, 0.045,
             r'$P^{(t)} \propto P_S(x \mid \theta_S^{\mathrm{Biased}}, c)\, P_E(x \mid \theta_E)$',
             ha='center', va='center', fontsize=24)

    ax_r10 = axs[2,1]
    cax = inset_axes(
        ax_r10, width="5.0%", height="120%", loc="lower left",
        bbox_to_anchor=(1.20, -0.00, 1, 1), bbox_transform=ax_r10.transAxes, borderpad=0.0
    )
    sm  = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[EVALUATOR_CMAP_ROUND])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Eval. Selection Probability', fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=12)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    name = f"scen{scenario_id}_grid_prior_r1_r10.png" if scenario_id is not None else "grid_prior_r1_r10.png"
    out  = os.path.join(outdir, name)
    fig.savefig(out, dpi=300, bbox_extra_artists=legend_artists + [cbar.ax, sep_line])
    print(f"[GRID-6] Saved: {out}")

    def _legend_handles(s_label, e_label):
        return [
            Line2D([0],[0], marker=PRIOR_SAMPLER_MARKER, linestyle='',
                   markerfacecolor=PRIOR_SAMPLER_FACE_HEX, markeredgecolor='white',
                   markeredgewidth=PRIOR_SAMPLER_EDGE_LW, markersize=10, label=s_label),
            Line2D([0],[0], marker=PRIOR_EVAL_MARKER, linestyle='',
                   markerfacecolor=PRIOR_EVAL_FACE_HEX, markeredgecolor='white',
                   markeredgewidth=PRIOR_EVAL_EDGE_LW, markersize=10, label=e_label),
        ]

    def _save_equation_png(eq_text: str, outpath: str,
                       fontsize=24, size=(6.8, 1.2), pad=0.02):
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, eq_text, ha="center", va="center", fontsize=fontsize)
        fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=pad, transparent=True)
        plt.close(fig)

    # Unbiased legend
    fig_lu = plt.figure(figsize=(3.2, 1.4))
    handles = _legend_handles(r'$P_S(x \mid \theta_S^{\mathrm{Unbiased}})$', r'$P_E(x \mid \theta_E)$')
    labels  = [r'$P_S(x \mid \theta_S^{\mathrm{Unbiased}})$', r'$P_E(x \mid \theta_E)$']
    lu = fig_lu.legend(handles, labels, loc="center", frameon=True, ncol=1, fontsize=24)
    p_unb = os.path.join(outdir, "legend_unbiased.png")
    fig_lu.savefig(p_unb, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig_lu)
    print(f"[LEGEND] Saved: {p_unb}")

    # Biased legend
    fig_lb = plt.figure(figsize=(3.2, 1.4))
    handles = _legend_handles(r'$P_S(x \mid \theta_S^{\mathrm{Biased}})$', r'$P_E(x \mid \theta_E)$')
    labels  = [r'$P_S(x \mid \theta_S^{\mathrm{Biased}})$', r'$P_E(x \mid \theta_E)$']
    lb = fig_lb.legend(handles, labels, loc="center", frameon=True, ncol=1, fontsize=24)

    p_bia = os.path.join(outdir, "legend_biased.png")
    fig_lb.savefig(p_bia, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig_lb)
    print(f"[LEGEND] Saved: {p_bia}")

    # Equation-only images (marker-less)
    eq_unb = r'$P^{(t)}(x) \propto P_S(x \mid \theta_S^{\mathrm{Unbiased}}, c)\, P_E(x \mid \theta_E)$'
    eq_bia = r'$P^{(t)}(x) \propto P_S(x \mid \theta_S^{\mathrm{Biased}}, c)\, P_E(x \mid \theta_E)$'

    p_eq_unb = os.path.join(outdir, "equation_unbiased.png")
    p_eq_bia = os.path.join(outdir, "equation_biased.png")

    _save_equation_png(eq_unb, p_eq_unb, fontsize=24, size=(6.8, 1.2))
    print(f"[EQUATION] Saved: {p_eq_unb}")

    _save_equation_png(eq_bia, p_eq_bia, fontsize=24, size=(6.8, 1.2))
    print(f"[EQUATION] Saved: {p_eq_bia}")

    # Colorbar only
    fig_cb = plt.figure(figsize=(0.9, 3.5))
    ax_cb  = fig_cb.add_axes([0.35, 0.05, 0.30, 0.90])
    cb2 = plt.colorbar(sm, cax=ax_cb, orientation="vertical")
    cb2.set_label('Eval. Selection Probability', fontsize=18)
    cb2.ax.tick_params(labelsize=16)
    p_cb = os.path.join(outdir, "colorbar_eval_prob.png")
    fig_cb.savefig(p_cb, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig_cb)
    print(f"[COLORBAR] Saved: {p_cb}")

    plt.close(fig)



def save_legend_png(handles, labels, outpath, ncols=1,
                    fontsize=24, frameon=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2.2*ncols, 1.2))
    leg = fig.legend(handles=handles, labels=labels, loc="center",
                     frameon=frameon, ncol=ncols, fontsize=fontsize)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

def save_colorbar_png(sm, outpath, orientation="vertical",
                      size=(0.9, 3.2), label="Eval. Selection Probability",
                      labelsize=18, ticksize=16):
    import matplotlib.pyplot as plt
    w, h = size
    fig = plt.figure(figsize=(w, h))
    if orientation == "vertical":
        ax = fig.add_axes([0.35, 0.05, 0.30, 0.90])  # [left, bottom, width, height]
    else:
        ax = fig.add_axes([0.05, 0.35, 0.90, 0.30])
    cb = plt.colorbar(sm, cax=ax, orientation=orientation)
    cb.set_label(label, fontsize=labelsize)
    cb.ax.tick_params(labelsize=ticksize)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Create 2D scatter grid (Prior / Round1 / Round10 × 2).")
    parser.add_argument("--scores_csv", required=True, help="Path to the CSV with 2D coordinate scores (e.g., two axes).")
    parser.add_argument("--logs", required=True, help="Path to the DIRECTORY with collaboration logs.")
    parser.add_argument("--outdir", default="plots_combined_normalized", help="Directory to save plots.")
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--grid", action="store_true", help="Make a 3x2 grid.")
    parser.add_argument("--other_logs", type=str, default=None, help="Log dir for the other condition.")
    parser.add_argument("--other_scores_csv", type=str, default=None, help="Optional 2D CSV for the other condition.")
    parser.add_argument("--eval_init_csv", type=str, default=None,
                        help="Evaluator prior CSV: (country,evaluator_init_prob) or (trial,country,score).")
    parser.add_argument("--tau_e", type=float, default=2.5,
                        help="Temperature for Boltzmann when using (trial,country,score).")
    parser.add_argument("--sampler_init_round", type=int, default=1,
                        help="Round to use as Sampler prior (default: 1).")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_scores_main = pd.read_csv(args.scores_csv)
    df_scores_main.drop_duplicates(subset=[df_scores_main.columns[0]], keep='first', inplace=True)
    df_scores_main.columns = [c.lower().strip() for c in df_scores_main.columns]

    log_paths = glob.glob(os.path.join(args.logs, "*.json"))
    if not log_paths:
        raise SystemExit(f"No '.json' files in {args.logs}")
    events_main = []
    for p in log_paths:
        with open(p, 'r', encoding='utf-8') as f:
            try:
                d = json.load(f)
                events_main.append(d['events'] if isinstance(d, dict) and 'events' in d else d)
            except json.JSONDecodeError:
                print(f"Warning: could not decode {p}")

    if FIXED_VMAX_OVERRIDE is not None:
        final_vmax = FIXED_VMAX_OVERRIDE
        print(f"Using fixed vmax override: {final_vmax}")
    else:
        def _maxprob_over(events_list, rounds):
            gmax = 0.0
            for r in [1, 10]:
                ep = _collect_probs(events_list, r, 'Evaluator')
                if ep: gmax = max(gmax, max(ep.values()))
            return gmax
        final_vmax = max(_maxprob_over(events_main, [1,10]), 1e-12)
        print(f"Auto vmax from logs: {final_vmax:.4f}")

    if args.grid:
        if not args.other_logs:
            raise SystemExit("--grid には --other_logs が必要です。")
        other_paths = glob.glob(os.path.join(args.other_logs, "*.json"))
        if not other_paths:
            raise SystemExit(f"No '.json' files in {args.other_logs}")
        events_other = []
        for p in other_paths:
            with open(p, 'r', encoding='utf-8') as f:
                try:
                    d = json.load(f)
                    events_other.append(d['events'] if isinstance(d, dict) and 'events' in d else d)
                except json.JSONDecodeError:
                    print(f"Warning: could not decode {p}")

        if args.other_scores_csv:
            df_scores_other = pd.read_csv(args.other_scores_csv)
            df_scores_other.drop_duplicates(subset=[df_scores_other.columns[0]], keep='first', inplace=True)
            df_scores_other.columns = [c.lower().strip() for c in df_scores_other.columns]
        else:
            df_scores_other = df_scores_main.copy()

        if args.eval_init_csv:
            eval_prior_unb = _load_eval_init_series(args.eval_init_csv, args.tau_e)
            eval_prior_bia = _load_eval_init_series(args.eval_init_csv, args.tau_e)
        else:
            eval_prior_unb = _load_eval_init_series(args.scores_csv, args.tau_e)
            eval_prior_bia = _load_eval_init_series(args.other_scores_csv or args.scores_csv, args.tau_e)

        vmax = FIXED_VMAX_OVERRIDE if FIXED_VMAX_OVERRIDE is not None else final_vmax
        generate_combined_grid6(
            df_scores_main, events_main, df_scores_other, events_other,
            eval_init_unb=eval_prior_unb, eval_init_bia=eval_prior_bia,
            outdir=args.outdir, scenario_id=args.id,
            vmin=0.0, vmax=vmax, sampler_init_round=args.sampler_init_round
        )
        return

if __name__ == "__main__":
    main()
