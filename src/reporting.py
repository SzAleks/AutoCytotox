"""
Reporting and Excel/Plot Export for the autocytotox Autogating Pipeline.

This module renders the comparison plots (scatter, Bland-Altman, residuals)
and writes the multi-sheet results Excel workbook produced at the end of a
pipeline run.

Author: Aleksander Szarzynski TUW 2026
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .data_loader import create_folder


def _setup_comparison_plot_dir(output_path: str) -> str:
    plots_path = os.path.join(output_path, 'comparison_plots')
    create_folder(plots_path)
    return plots_path


def plot_comparison_scatter(comparison_df, summary, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['cytotoxicity_mean', 'manual_mean'])
    if len(valid) < 3:
        return

    summary = summary or {}
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'Opt': '#1f77b4', 'Ref': '#d62728'}
    for label, group in valid.groupby('opt_ref'):
        ax.scatter(
            group['manual_mean'],
            group['cytotoxicity_mean'],
            c=colors.get(label, 'gray'),
            label=label,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5,
        )

    all_vals = np.concatenate([valid['manual_mean'].values, valid['cytotoxicity_mean'].values])
    lo = min(all_vals.min(), 0)
    hi = max(all_vals.max() * 1.1, 10)
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='Identity')

    slope, intercept, _, _, _ = stats.linregress(valid['manual_mean'], valid['cytotoxicity_mean'])
    x_fit = np.linspace(lo, hi, 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'g-', alpha=0.6, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    r_val = summary.get('pearson_r', np.nan)
    mae_val = summary.get('mae', np.nan)
    rmse_val = summary.get('rmse', np.nan)
    ax.set_xlabel('Manual Cytotoxicity (%)', fontsize=12)
    ax.set_ylabel('Autogating Cytotoxicity (%)', fontsize=12)
    ax.set_title(
        f'Autogating vs Manual Analysis\n'
        f'r={r_val:.3f}, MAE={mae_val:.2f}%, RMSE={rmse_val:.2f}%',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'comparison_scatter.{fmt}'), dpi=150)
    plt.close()


def plot_bland_altman(comparison_df, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['cytotoxicity_mean', 'manual_mean'])
    if len(valid) < 3:
        return

    mean_vals = (valid['cytotoxicity_mean'] + valid['manual_mean']) / 2
    diff_vals = valid['cytotoxicity_mean'] - valid['manual_mean']
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Opt': '#1f77b4', 'Ref': '#d62728'}
    for label, group in valid.groupby('opt_ref'):
        idx = group.index
        ax.scatter(
            mean_vals.loc[idx],
            diff_vals.loc[idx],
            c=colors.get(label, 'gray'),
            label=label,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5,
        )

    ax.axhline(y=mean_diff, color='green', linestyle='-', label=f'Mean diff: {mean_diff:.2f}%')
    ax.axhline(y=mean_diff + 1.96 * std_diff, color='red', linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}%')
    ax.axhline(y=mean_diff - 1.96 * std_diff, color='red', linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}%')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)

    ax.set_xlabel('Mean of Auto & Manual (%)', fontsize=12)
    ax.set_ylabel('Difference (Auto − Manual) (%)', fontsize=12)
    ax.set_title('Bland-Altman Plot', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'bland_altman.{fmt}'), dpi=150)
    plt.close()


def plot_cv_comparison(comparison_df, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['cv_percent', 'manual_cv'])
    if len(valid) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for opt_ref, group in valid.groupby('opt_ref'):
        day_cv = group.groupby('day').agg(
            auto_cv_mean=('cv_percent', 'mean'),
            manual_cv_mean=('manual_cv', 'mean'),
        ).reset_index()
        marker = 'o' if opt_ref == 'Opt' else 's'
        axes[0].plot(day_cv['day'], day_cv['auto_cv_mean'], f'-{marker}', label=f'Auto {opt_ref}')
        axes[0].plot(day_cv['day'], day_cv['manual_cv_mean'], f'--{marker}', label=f'Manual {opt_ref}')

    axes[0].set_xlabel('Day', fontsize=11)
    axes[0].set_ylabel('CV%', fontsize=11)
    axes[0].set_title('CV% Over Time', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(range(9))

    axes[1].scatter(valid['manual_cv'], valid['cv_percent'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    lims = [0, max(valid[['cv_percent', 'manual_cv']].max().max() * 1.1, 10)]
    axes[1].plot(lims, lims, 'k--', alpha=0.4)
    axes[1].set_xlabel('Manual CV%', fontsize=11)
    axes[1].set_ylabel('Autogating CV%', fontsize=11)
    axes[1].set_title('CV% Auto vs Manual', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'cv_comparison.{fmt}'), dpi=150)
    plt.close()


def plot_day_comparison(comparison_df, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['cytotoxicity_mean', 'manual_mean'])
    if len(valid) < 2:
        return

    conditions = sorted(valid['condition'].unique())
    n_cond = len(conditions)

    for opt_ref in ['Opt', 'Ref']:
        fig, axes = plt.subplots(1, max(n_cond, 1), figsize=(6 * n_cond, 5), squeeze=False)
        for index, cond in enumerate(conditions):
            ax = axes[0, index]
            subset = valid[(valid['opt_ref'] == opt_ref) & (valid['condition'] == cond)].sort_values('day')
            if subset.empty:
                ax.set_title(f'{opt_ref} − {cond}: No data')
                continue

            ax.errorbar(subset['day'], subset['cytotoxicity_mean'], yerr=subset['cytotoxicity_std'], fmt='o-', label='Autogating', capsize=3, markersize=5)
            ax.errorbar(subset['day'], subset['manual_mean'], yerr=subset['manual_std'], fmt='s--', label='Manual', capsize=3, markersize=5)
            ax.set_xlabel('Day', fontsize=11)
            ax.set_ylabel('Cytotoxicity (%)', fontsize=11)
            ax.set_title(f'{opt_ref} − Condition {cond}', fontsize=12)
            ax.legend(fontsize=9)
            ax.set_xticks(range(9))

        plt.suptitle(f'{opt_ref}: Day-over-Day Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'day_comparison_{opt_ref}.{fmt}'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_heatmap_diff(comparison_df, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['diff'])
    if len(valid) < 2:
        return

    for opt_ref in ['Opt', 'Ref']:
        subset = valid[valid['opt_ref'] == opt_ref]
        if subset.empty:
            continue

        pivot = subset.pivot_table(index='day', columns='condition', values='diff', aggfunc='mean')
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 8))
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 5)
        import seaborn as sns
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            cbar_kws={'label': 'Auto − Manual (%)'},
        )
        ax.set_title(f'{opt_ref}: Difference Heatmap (Auto − Manual)', fontsize=12)
        ax.set_ylabel('Day')
        ax.set_xlabel('Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'diff_heatmap_{opt_ref}.{fmt}'), dpi=150)
        plt.close()


def plot_precision_profile(comparison_df, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['cytotoxicity_mean', 'manual_mean'])
    if len(valid) < 4:
        return

    mean_level = (valid['cytotoxicity_mean'] + valid['manual_mean']) / 2
    abs_err = (valid['cytotoxicity_mean'] - valid['manual_mean']).abs()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = {'Opt': '#1f77b4', 'Ref': '#d62728'}
    for label, group in valid.groupby('opt_ref'):
        idx = group.index
        ax.scatter(mean_level.loc[idx], abs_err.loc[idx], c=colors.get(label, 'gray'), label=label, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    if len(mean_level) >= 6:
        sorted_idx = mean_level.sort_values().index
        sorted_level = mean_level.loc[sorted_idx].values
        sorted_err = abs_err.loc[sorted_idx].values
        n_bins = min(8, len(sorted_level) // 2)
        if n_bins >= 2:
            bin_edges = np.linspace(sorted_level.min(), sorted_level.max(), n_bins + 1)
            bin_centers = []
            bin_medians = []
            for bin_index in range(n_bins):
                mask = (sorted_level >= bin_edges[bin_index]) & (sorted_level < bin_edges[bin_index + 1])
                if bin_index == n_bins - 1:
                    mask = (sorted_level >= bin_edges[bin_index]) & (sorted_level <= bin_edges[bin_index + 1])
                if mask.sum() > 0:
                    bin_centers.append(sorted_level[mask].mean())
                    bin_medians.append(np.median(sorted_err[mask]))
            ax.plot(bin_centers, bin_medians, 'k-o', markersize=4, linewidth=1.5, label='Binned median', zorder=5)

    ax.set_xlabel('Mean Cytotoxicity (%)', fontsize=11)
    ax.set_ylabel('|Auto - Manual| (%)', fontsize=11)
    ax.set_title('A: Precision Profile (Absolute Error)', fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    cv_valid = valid.dropna(subset=['cv_percent'])
    if len(cv_valid) >= 4:
        for label, group in cv_valid.groupby('opt_ref'):
            ax.scatter(group['cytotoxicity_mean'], group['cv_percent'], c=colors.get(label, 'gray'), label=label, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, label='CV% = 20%')
        ax.set_xlabel('Autogating Cytotoxicity (%)', fontsize=11)
        ax.set_ylabel('CV%', fontsize=11)
        ax.set_title('B: CV% vs Cytotoxicity Level', fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'precision_profile.{fmt}'), dpi=150)
    plt.close()


def plot_residual_analysis(comparison_df, output_path, fmt='png'):
    valid = comparison_df.dropna(subset=['cytotoxicity_mean', 'manual_mean'])
    if len(valid) < 4:
        return

    residuals = valid['cytotoxicity_mean'].values - valid['manual_mean'].values
    fitted = valid['manual_mean'].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(fitted, residuals, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    if len(fitted) >= 4:
        slope, intercept, _, p_val, _ = stats.linregress(fitted, residuals)
        x_line = np.linspace(fitted.min(), fitted.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, 'g-', alpha=0.7, label=f'Trend: slope={slope:.3f}, p={p_val:.3f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Manual Cytotoxicity (%)', fontsize=11)
    ax.set_ylabel('Residual (Auto - Manual)', fontsize=11)
    ax.set_title('A: Residuals vs Fitted', fontsize=12)

    ax = axes[1]
    sorted_res = np.sort(residuals)
    n_points = len(sorted_res)
    theoretical = stats.norm.ppf(np.arange(1, n_points + 1) / (n_points + 1))
    ax.scatter(theoretical, sorted_res, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    q25, q75 = np.percentile(sorted_res, [25, 75])
    t25, t75 = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    if t75 != t25:
        slope_qq = (q75 - q25) / (t75 - t25)
        intercept_qq = q25 - slope_qq * t25
        x_qq = np.array([theoretical.min(), theoretical.max()])
        ax.plot(x_qq, slope_qq * x_qq + intercept_qq, 'r--', alpha=0.7)
    if 3 <= n_points <= 5000:
        _, sw_p = stats.shapiro(residuals)
        ax.set_title(f'B: Q-Q Plot (Shapiro p={sw_p:.3f})', fontsize=12)
    else:
        ax.set_title('B: Q-Q Plot', fontsize=12)
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)

    ax = axes[2]
    ax.hist(residuals, bins=max(5, n_points // 3), edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=residuals.mean(), color='green', linestyle='-', alpha=0.7, label=f'Mean: {residuals.mean():.2f}')
    ax.set_xlabel('Residual (Auto - Manual)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('C: Residual Distribution', fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'residual_analysis.{fmt}'), dpi=150)
    plt.close()


def render_comparison_plots(comparison_df, summary_metrics, output_path, plot_format='png') -> str:
    plots_path = _setup_comparison_plot_dir(output_path)
    plot_comparison_scatter(comparison_df, summary_metrics, plots_path, plot_format)
    plot_bland_altman(comparison_df, plots_path, plot_format)
    plot_cv_comparison(comparison_df, plots_path, plot_format)
    plot_day_comparison(comparison_df, plots_path, plot_format)
    plot_heatmap_diff(comparison_df, plots_path, plot_format)
    plot_precision_profile(comparison_df, plots_path, plot_format)
    plot_residual_analysis(comparison_df, plots_path, plot_format)
    return plots_path


def save_comprehensive_excel(
    output_path: str,
    combined_results: pd.DataFrame,
    auto_agg: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame],
    summary_metrics: Optional[Dict],
    manual_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    prefix: str = 'autocytotox',
) -> str:
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    filename = f'{prefix}_results_{args.method}_{int(args.percent * 100)}_{timestamp}.xlsx'
    filepath = os.path.join(output_path, filename)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        combined_results.to_excel(writer, sheet_name='Raw_Results', index=False)
        auto_agg.to_excel(writer, sheet_name='Aggregated', index=False)

        params_data = {
            'parameter': [
                'method', 'percent', 'bw_adjust', 'min_distance', 'gate2_method',
                'gate_derivation', 'percentile_q', 'k_sigma', 'beads_x_cutoff',
                'effector_cell', 'target_cell', 'mwe', 'mwe_n', 'data_path', 'timestamp',
            ],
            'value': [
                args.method, args.percent, args.bw_adjust, args.min_distance,
                getattr(args, 'gate2_method', 'knn'),
                getattr(args, 'gate_derivation', 'max'),
                getattr(args, 'percentile_q', 0.99),
                getattr(args, 'k_sigma', 2.5),
                args.beads_x_cutoff, args.effector_cell, args.target_cell,
                args.mwe, args.mwe_n if args.mwe else 'N/A', args.data_path, timestamp,
            ],
        }
        pd.DataFrame(params_data).to_excel(writer, sheet_name='Parameters', index=False)

        if comparison_df is not None and not comparison_df.empty:
            comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

        if summary_metrics is not None and summary_metrics.get('n_comparisons', 0) > 0:
            pd.DataFrame([summary_metrics]).to_excel(writer, sheet_name='Summary_Metrics', index=False)

        if manual_df is not None and not manual_df.empty:
            manual_df.to_excel(writer, sheet_name='Manual_Data', index=False)

    return filepath