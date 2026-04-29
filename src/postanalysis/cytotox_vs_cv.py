#!/usr/bin/env python3
"""
Cytotoxicity-vs-CV precision analysis for AutoCytotox.

- manual CV is computed per user and then averaged by group
- autogating CV is matched to the same group
- CV advantage is associated with cytotoxicity using Spearman correlation
- cytotoxicity thresholds from 10% to 75% are scanned with Wilcoxon
  signed-rank tests for an autogating precision advantage

Author: Aleksander Szarzynski TUW 2026
"""

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
from scipy import stats  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

from .. import paths as config
from ..cytotox_evaluator import ChannelConfig, GatingConfig, PlotConfig
from ..data_loader import create_folder, get_subfolders
from ..identifiers import identifier_by_well_row
from ..workflow import (
    DEFAULT_ANALYSIS_MAX_CV,
    DEFAULT_ANALYSIS_MIN_CYTOTOX,
    add_analysis_cutoff_flags,
    aggregate_autogating_results,
    filter_analysis_cutoffs,
    parse_manual_excel,
    process_folders,
)


GROUP_COLS = ['day', 'condition', 'opt_ref']
CV_CUTOFF = 50.0


# ============================================================================
# Config extraction
# ============================================================================

def _config_key(cfg: Dict) -> str:
    """Deterministic string key for a gridsearch config dict."""
    gate_derivation = cfg.get('gate_derivation', 'max')
    gate2_method = cfg.get('gate2_method', 'knn')
    gd_tag = ''
    if gate_derivation == 'percentile':
        gd_tag = f"_gd-pct{cfg.get('percentile_q', 0.99)}"
    elif gate_derivation == 'mean_ksd':
        gd_tag = f"_gd-ksd{cfg.get('k_sigma', 2.5)}"
    elif gate_derivation != 'max':
        gd_tag = f"_gd-{gate_derivation}"
    gate2_tag = '' if gate2_method == 'knn' else f"_g2-{gate2_method}"
    return (f"{cfg['method']}_p{cfg['percent']}_"
            f"bw{cfg['bw_adjust']}_md{cfg['min_distance']}{gate2_tag}{gd_tag}")


def extract_top_configs(source, n: int = 1) -> List[Dict]:
    """
    Extract the top ranked gridsearch configurations.

    The cytotoxicity-vs-CV analysis uses the operational best configuration by
    default, but the helper accepts ``n`` for callers that need to inspect the
    ranking table.
    """
    if isinstance(source, str):
        try:
            df = pd.read_excel(source, sheet_name='Top_10')
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            df = pd.read_excel(source, sheet_name='Results')
    else:
        df = source.copy()

    if 'status' in df.columns:
        df = df[df['status'] == 'OK']
    if 'composite_rank' in df.columns:
        df = df[df['composite_rank'].notna()]
    if 'composite_score' in df.columns:
        df = df[df['composite_score'].notna() & (df['composite_score'] > 0)]
    for metric in ['pearson_r', 'mean_cv']:
        if metric in df.columns:
            df = df[df[metric].notna()]
            break

    if 'composite_rank' in df.columns:
        df = df.sort_values('composite_rank')

    configs: List[Dict[str, Any]] = []
    seen_keys = set()
    for _, row in df.iterrows():
        if len(configs) >= n:
            break

        cfg = {
            'method': row['method'],
            'percent': float(row['percent']),
            'bw_adjust': float(row['bw_adjust']),
            'min_distance': int(row['min_distance']),
            'gate2_method': row.get('gate2_method', 'knn'),
            'gate_derivation': row.get('gate_derivation', 'max'),
        }
        gate_derivation = cfg['gate_derivation']
        if gate_derivation == 'percentile' and pd.notna(row.get('percentile_q')):
            cfg['percentile_q'] = float(row['percentile_q'])
        elif gate_derivation == 'mean_ksd' and pd.notna(row.get('k_sigma')):
            cfg['k_sigma'] = float(row['k_sigma'])

        key = _config_key(cfg)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        method_tag = cfg['method'][:4]
        percent_tag = int(cfg['percent'] * 100)
        label = f"{method_tag}_p{percent_tag}_bw{cfg['bw_adjust']}"
        if cfg.get('gate2_method', 'knn') != 'knn':
            label += f"_{cfg['gate2_method']}"
        if gate_derivation != 'max':
            label += f"_{gate_derivation[:3]}"

        cfg['label'] = label
        cfg['rank'] = len(configs) + 1
        cfg['config_key'] = key
        configs.append(cfg)

    return configs


# ============================================================================
# Data loading
# ============================================================================

def _load_auto_agg_from_cache(gridsearch_dir: str, config_key: str) -> Optional[pd.DataFrame]:
    """Load auto_agg.csv from direct or legacy gridsearch output folders."""
    candidates = [
        os.path.join(gridsearch_dir, config_key, 'auto_agg.csv'),
        os.path.join(gridsearch_dir, 'configs', config_key, 'auto_agg.csv'),
    ]
    for csv_path in candidates:
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    return None


def compute_manual_per_analyst_cv(manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CV percent across biological replicates for each user and group.

    Manual precision is later averaged across users, matching the described
    method comparison.
    """
    if manual_df.empty:
        return pd.DataFrame()

    records = []
    for analyst in sorted(manual_df['analyst'].unique()):
        adf = manual_df[manual_df['analyst'] == analyst]
        grouped = adf.groupby(GROUP_COLS).agg(
            manual_mean=('mean_tech', 'mean'),
            manual_std=('mean_tech', 'std'),
            n_bio_reps=('mean_tech', 'count'),
        ).reset_index()
        grouped['manual_cv'] = grouped.apply(
            lambda row: (row['manual_std'] / row['manual_mean'] * 100)
            if pd.notna(row['manual_mean']) and row['manual_mean'] != 0
            else np.nan,
            axis=1,
        )
        grouped['analyst'] = analyst
        records.append(grouped)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _run_single_autogating(cfg: Dict, data_path: str, output_path: str) -> pd.DataFrame:
    """Run one autogating config when cached aggregation is unavailable."""
    all_folders = get_subfolders(data_path)
    day_folders = sorted(folder for folder in all_folders if folder.startswith('Day '))

    args = argparse.Namespace(
        data_path=data_path,
        output_path=output_path,
        method=cfg['method'],
        percent=cfg['percent'],
        bw_adjust=cfg['bw_adjust'],
        min_distance=cfg['min_distance'],
        gate2_method=cfg.get('gate2_method', 'knn'),
        gate_derivation=cfg.get('gate_derivation', 'max'),
        percentile_q=cfg.get('percentile_q', 0.99),
        k_sigma=cfg.get('k_sigma', 2.5),
        beads_x_cutoff=2.4e6,
        effector_cell='NK',
        target_cell='K562',
        verbose=False,
        subsample=0.0,
        plot_format='png',
    )

    channels = ChannelConfig()
    gating = GatingConfig(
        percent=cfg['percent'],
        bw_adjust=cfg['bw_adjust'],
        min_distance=cfg['min_distance'],
        beads_x_cutoff=2.4e6,
        subsample=0.0,
        gate2_method=cfg.get('gate2_method', 'knn'),
        gate_derivation=cfg.get('gate_derivation', 'max'),
        percentile_q=cfg.get('percentile_q', 0.99),
        k_sigma=cfg.get('k_sigma', 2.5),
    )
    plotting = PlotConfig(enable_all=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        combined = process_folders(
            day_folders,
            args,
            channels,
            gating,
            plotting,
            identifier_by_well_row,
        )

    if combined.empty:
        raise RuntimeError(f"Autogating produced no results for {cfg.get('label', '?')}")

    return aggregate_autogating_results(combined)


def _filter_auto_agg(
    auto_agg: pd.DataFrame,
    min_cytotox: float,
    max_cv: float,
    apply_cutoffs: bool,
) -> pd.DataFrame:
    flagged = add_analysis_cutoff_flags(
        auto_agg,
        value_col='cytotoxicity_mean',
        cv_col='cv_percent',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    return filter_analysis_cutoffs(
        flagged,
        value_col='cytotoxicity_mean',
        cv_col='cv_percent',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )


def paired_precision_table(
    manual_cv_df: pd.DataFrame,
    auto_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match averaged manual CV and autogating CV by experimental group.

    Positive ``cv_advantage`` means autogating has a lower CV percent than the
    averaged manual evaluation.
    """
    if manual_cv_df.empty or auto_agg.empty:
        return pd.DataFrame()

    manual_avg = manual_cv_df.groupby(GROUP_COLS).agg(
        manual_mean=('manual_mean', 'mean'),
        manual_cv=('manual_cv', 'mean'),
        n_users=('analyst', 'nunique'),
    ).reset_index()

    paired = pd.merge(
        auto_agg[[*GROUP_COLS, 'cytotoxicity_mean', 'cv_percent']],
        manual_avg,
        on=GROUP_COLS,
        how='inner',
    ).dropna(subset=['cytotoxicity_mean', 'cv_percent', 'manual_mean', 'manual_cv'])

    if paired.empty:
        return paired

    paired = paired.rename(columns={'cv_percent': 'auto_cv'})
    paired['cytotox_ref'] = (
        paired['cytotoxicity_mean'] + paired['manual_mean']
    ) / 2
    paired['cv_advantage'] = paired['manual_cv'] - paired['auto_cv']
    return paired


# ============================================================================
# Statistics and plotting
# ============================================================================

def precision_statistics(paired: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run Spearman association and Wilcoxon threshold scans."""
    lines = []
    lines.append('Cytotoxicity-vs-CV precision analysis')
    lines.append('=' * 45)

    if paired.empty:
        empty = pd.DataFrame()
        lines.append('No paired manual/autogating precision values available.')
        return empty, empty, '\n'.join(lines)

    lines.append(f"Paired groups: {len(paired)}")
    lines.append(f"Mean manual CV%: {paired['manual_cv'].mean():.3f}")
    lines.append(f"Mean autogating CV%: {paired['auto_cv'].mean():.3f}")
    lines.append(
        f"Mean CV advantage (manual - auto): {paired['cv_advantage'].mean():+.3f}"
    )

    if len(paired) >= 3:
        rho, p_value = stats.spearmanr(paired['cytotox_ref'], paired['cv_advantage'])
    else:
        rho, p_value = np.nan, np.nan
    spearman_df = pd.DataFrame([{
        'test': 'Spearman correlation',
        'x': 'cytotoxicity',
        'y': 'manual_cv_minus_auto_cv',
        'rho': float(rho) if np.isfinite(rho) else np.nan,
        'p_value': float(p_value) if np.isfinite(p_value) else np.nan,
        'n': int(len(paired)),
    }])

    lines.append('')
    lines.append('Spearman association')
    lines.append(
        f"rho={spearman_df.loc[0, 'rho']:.4f}, "
        f"p={spearman_df.loc[0, 'p_value']:.4g}"
    )

    breakpoint_records = []
    for threshold in range(10, 80, 5):
        subset = paired[paired['cytotox_ref'] >= threshold]
        diff = subset['cv_advantage'].dropna().to_numpy(dtype=float)
        if len(diff) >= 5:
            try:
                stat_w, p_w = stats.wilcoxon(diff, alternative='greater')
                stat_w = float(stat_w)
                p_w = float(p_w)
            except ValueError:
                stat_w, p_w = np.nan, np.nan
        else:
            stat_w, p_w = np.nan, np.nan

        breakpoint_records.append({
            'threshold_percent': float(threshold),
            'n_above_threshold': int(len(diff)),
            'mean_cv_advantage': float(np.mean(diff)) if len(diff) else np.nan,
            'median_cv_advantage': float(np.median(diff)) if len(diff) else np.nan,
            'wilcoxon_statistic': stat_w,
            'p_value_one_sided_auto_better': p_w,
            'significant_005': bool(p_w < 0.05) if np.isfinite(p_w) else False,
        })

    breakpoint_df = pd.DataFrame(breakpoint_records)
    tested = breakpoint_df.dropna(subset=['p_value_one_sided_auto_better'])

    lines.append('')
    lines.append('Wilcoxon threshold scan')
    if tested.empty:
        lines.append('No threshold had enough paired values for Wilcoxon testing.')
    else:
        best = tested.sort_values('p_value_one_sided_auto_better').iloc[0]
        lines.append(
            f"Best threshold: cytotoxicity >= {best['threshold_percent']:.0f}%"
        )
        lines.append(f"n above threshold: {best['n_above_threshold']:.0f}")
        lines.append(f"mean CV advantage: {best['mean_cv_advantage']:+.3f}")
        lines.append(
            f"Wilcoxon p={best['p_value_one_sided_auto_better']:.4g} "
            "(one-sided: autogating lower CV%)"
        )

    return spearman_df, breakpoint_df, '\n'.join(lines)


def plot_cytotox_vs_cv(
    manual_cv_df: pd.DataFrame,
    auto_agg: pd.DataFrame,
    output_path: str,
    cv_cutoff: float = CV_CUTOFF,
    fmt: str = 'png',
) -> str:
    """Plot manual user CV and autogating CV against cytotoxicity."""
    create_folder(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'User 1': '#e41a1c', 'User 2': '#377eb8', 'User 3': '#4daf4a'}

    for analyst in sorted(manual_cv_df['analyst'].unique()):
        sub = manual_cv_df[manual_cv_df['analyst'] == analyst].dropna(
            subset=['manual_mean', 'manual_cv'],
        )
        ax.scatter(
            sub['manual_mean'],
            sub['manual_cv'],
            color=colors.get(analyst, 'gray'),
            alpha=0.45,
            s=30,
            edgecolors='none',
            label=f'{analyst} (n={len(sub)})',
        )

    auto_valid = auto_agg.dropna(subset=['cytotoxicity_mean', 'cv_percent'])
    ax.scatter(
        auto_valid['cytotoxicity_mean'],
        auto_valid['cv_percent'],
        color='#ff7f00',
        alpha=0.75,
        s=46,
        edgecolors='black',
        linewidth=0.4,
        marker='D',
        label=f'Autogating (n={len(auto_valid)})',
    )

    ax.set_xlabel('Cytotoxicity (%)')
    ax.set_ylabel('CV%')
    ax.set_title('Cytotoxicity vs CV%')
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, cv_cutoff)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_path, f'cytotox_vs_cv.{fmt}')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


# ============================================================================
# Main entry point
# ============================================================================

def run_cytotox_vs_cv(
    config_entry: Dict,
    manual_excel: str,
    output_path: str,
    data_path: Optional[str] = None,
    gridsearch_dir: Optional[str] = None,
    plot_format: str = 'png',
    min_cytotox: float = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: float = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run the manuscript-aligned cytotoxicity-vs-CV analysis."""
    create_folder(output_path)

    if verbose:
        print('Loading manual data...')
    manual_df = parse_manual_excel(manual_excel)
    manual_cv_df = add_analysis_cutoff_flags(
        compute_manual_per_analyst_cv(manual_df),
        value_col='manual_mean',
        cv_col='manual_cv',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if 'passes_analysis_cutoffs' in manual_cv_df.columns:
        manual_cv_df = manual_cv_df.loc[manual_cv_df['passes_analysis_cutoffs']].copy()

    if verbose:
        print(f"  Retained manual user-group CV rows: {len(manual_cv_df)}")

    label = config_entry.get('label', 'autogating')
    key = config_entry.get('config_key', _config_key(config_entry))
    auto_agg = None
    if gridsearch_dir:
        auto_agg = _load_auto_agg_from_cache(gridsearch_dir, key)
        if auto_agg is not None and verbose:
            print(f'Loaded cached autogating aggregation: {label}')

    if auto_agg is None:
        if data_path is None:
            if verbose:
                print('No cached autogating aggregation and no data_path supplied.')
            return {}
        if verbose:
            print(f'Running autogating config: {label}')
        run_output = os.path.join(output_path, f'_run_{label}')
        create_folder(run_output)
        auto_agg = _run_single_autogating(config_entry, data_path, run_output)

    auto_agg = _filter_auto_agg(
        auto_agg,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if verbose:
        print(f'  Retained autogating groups: {len(auto_agg)}')

    paired_df = paired_precision_table(manual_cv_df, auto_agg)
    spearman_df, breakpoint_df, report = precision_statistics(paired_df)

    plot_path = plot_cytotox_vs_cv(
        manual_cv_df,
        auto_agg,
        output_path,
        fmt=plot_format,
    )

    paired_path = os.path.join(output_path, 'paired_precision.csv')
    spearman_path = os.path.join(output_path, 'spearman_precision.csv')
    breakpoint_path = os.path.join(output_path, 'breakpoint_wilcoxon.csv')
    report_path = os.path.join(output_path, 'cytotox_vs_cv_report.txt')

    if not paired_df.empty:
        paired_df.to_csv(paired_path, index=False)
    if not spearman_df.empty:
        spearman_df.to_csv(spearman_path, index=False)
    if not breakpoint_df.empty:
        breakpoint_df.to_csv(breakpoint_path, index=False)
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(report)

    if verbose:
        print(report)
        print(f'\nPlot saved to: {plot_path}')
        print(f'Report saved to: {report_path}')

    return {
        'manual_cv_df': manual_cv_df,
        'auto_agg': auto_agg,
        'paired_precision_df': paired_df,
        'spearman_df': spearman_df,
        'breakpoint_df': breakpoint_df,
        'report': report,
        'plot_path': plot_path,
        'report_path': report_path,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Cytotoxicity-vs-CV precision analysis for the best autogating config.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--gridsearch-dir',
        type=str,
        default=None,
        help='Path to gridsearch output directory with cached auto_agg and gridsearch Excel.',
    )
    parser.add_argument(
        '--gridsearch-excel',
        type=str,
        default=None,
        help='Path to gridsearch results Excel.',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=os.path.join(config.PATH_RAW_DATA, 'data', 'EX27_180924_modified'),
        help='Path to FCS data directory, needed only if cache is unavailable.',
    )
    parser.add_argument(
        '--manual-excel',
        type=str,
        default=os.path.join(
            config.PATH_RAW_DATA,
            'output',
            'manual_analysis',
            'Cytotox_analysis_User_1_2_3_copy.xlsx',
        ),
        help='Path to manual analysis Excel.',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output directory. Defaults to gridsearch_dir/cytotox_vs_cv.',
    )
    parser.add_argument(
        '--plot-format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
    )
    parser.add_argument(
        '--min-analysis-cytotox',
        type=float,
        default=DEFAULT_ANALYSIS_MIN_CYTOTOX,
    )
    parser.add_argument(
        '--max-analysis-cv',
        type=float,
        default=DEFAULT_ANALYSIS_MAX_CV,
    )
    parser.add_argument('--disable-analysis-cutoffs', action='store_true')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args(argv)

    if args.gridsearch_dir is None and args.gridsearch_excel is None:
        parser.error('Either --gridsearch-dir or --gridsearch-excel is required')

    args.apply_cutoffs = not args.disable_analysis_cutoffs

    if args.gridsearch_excel is None and args.gridsearch_dir:
        candidates = sorted([
            filename for filename in os.listdir(args.gridsearch_dir)
            if filename.startswith('autocytotox_gridsearch_')
            and filename.endswith('.xlsx')
            and 'rankfixed' not in filename
        ])
        if not candidates:
            parser.error(f'No autocytotox_gridsearch_*.xlsx found in {args.gridsearch_dir}')
        args.gridsearch_excel = os.path.join(args.gridsearch_dir, candidates[-1])

    if args.output_path is None:
        if args.gridsearch_dir:
            args.output_path = os.path.join(args.gridsearch_dir, 'cytotox_vs_cv')
        else:
            args.output_path = os.path.join(
                os.path.dirname(args.gridsearch_excel),
                'cytotox_vs_cv',
            )

    return args


def main(argv=None):
    args = parse_args(argv)

    print('=' * 60)
    print('Cytotoxicity-vs-CV Precision Analysis')
    print('=' * 60)

    configs = extract_top_configs(args.gridsearch_excel, n=1)
    if not configs:
        print('ERROR: No valid configs found in gridsearch results.')
        sys.exit(1)

    best_config = configs[0]
    print(f"Best config: #{best_config['rank']} {best_config['label']}")

    run_cytotox_vs_cv(
        config_entry=best_config,
        manual_excel=args.manual_excel,
        output_path=args.output_path,
        data_path=args.data_path,
        gridsearch_dir=args.gridsearch_dir,
        plot_format=args.plot_format,
        min_cytotox=args.min_analysis_cytotox,
        max_cv=args.max_analysis_cv,
        apply_cutoffs=args.apply_cutoffs,
        verbose=not args.quiet,
    )

    print(f'\n{"=" * 60}')
    print('CYTOTOX-VS-CV ANALYSIS COMPLETE')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
