"""
High-Level Workflow Composition for the autocytotox Autogating Pipeline.

This module defines the orchestration glue that walks the discovered Day
folders, runs the gating engine on each one, aggregates per-well
cytotoxicity into per-group means and CV%, parses the optional manual
analyst Excel, and computes the comparison metrics.

Author: Aleksander Szarzynski TUW 2026
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..analytics import concordance_correlation_coefficient
from ..cytotox_evaluator import ChannelConfig, CytotoxEvaluator, GatingConfig, PlotConfig
from ..data_loader import get_subfolders
from ..identifiers import IdentifierHandler, identifier_by_well_row


DEFAULT_ANALYSIS_MIN_CYTOTOX = 5.0
DEFAULT_ANALYSIS_MAX_CV = 25.0

ANALYST_SHEETS = ['User 1', 'User 2', 'User 3']


def identifier_fetcher(filename_list: List[str]) -> List[str]:
    return identifier_by_well_row(filename_list)


def parse_manual_excel(excel_path: str) -> pd.DataFrame:
    condition_cols = {
        'B': {'tech1': 4, 'tech2': 5, 'tech3': 6, 'mean': 7, 'std': 8},
        'F': {'tech1': 12, 'tech2': 13, 'tech3': 14, 'mean': 15, 'std': 16},
        'R': {'tech1': 20, 'tech2': 21, 'tech3': 22, 'mean': 23, 'std': 24},
    }

    bio_rep_starts = {
        ('Opt', 1): 2, ('Opt', 2): 12, ('Opt', 3): 22,
        ('Ref', 1): 33, ('Ref', 2): 43, ('Ref', 3): 53,
    }

    records = []
    for analyst in ANALYST_SHEETS:
        try:
            df = pd.read_excel(excel_path, analyst, header=None)
        except Exception as exc:
            print(f"  Warning: Could not read sheet '{analyst}': {exc}")
            continue

        for (opt_ref, bio_rep), start_row in bio_rep_starts.items():
            for day_offset in range(9):
                row_idx = start_row + day_offset
                if row_idx >= len(df):
                    continue

                for cond, cols in condition_cols.items():
                    tech_vals = []
                    for tech_col in ['tech1', 'tech2', 'tech3']:
                        value = df.iloc[row_idx, cols[tech_col]]
                        tech_vals.append(float(value) if pd.notna(value) else np.nan)

                    mean_val = df.iloc[row_idx, cols['mean']]
                    std_val = df.iloc[row_idx, cols['std']]

                    records.append({
                        'analyst': analyst,
                        'opt_ref': opt_ref,
                        'bio_rep': bio_rep,
                        'day': day_offset,
                        'condition': cond,
                        'tech1': tech_vals[0],
                        'tech2': tech_vals[1],
                        'tech3': tech_vals[2],
                        'mean_tech': float(mean_val) if pd.notna(mean_val) else np.nan,
                        'std_tech': float(std_val) if pd.notna(std_val) else np.nan,
                    })

    return pd.DataFrame(records)


def parse_folder_metadata(folder_name: str) -> Tuple[int, str]:
    parts = folder_name.rsplit('_', 1)
    day_str = parts[0].replace('Day ', '')
    day = int(day_str)
    condition = parts[1] if len(parts) > 1 else 'unknown'
    return day, condition


def aggregate_autogating_results(combined_results: pd.DataFrame) -> pd.DataFrame:
    if combined_results.empty or 'Folder' not in combined_results.columns:
        return pd.DataFrame()

    records = []
    for folder_name, folder_df in combined_results.groupby('Folder'):
        try:
            day, condition = parse_folder_metadata(folder_name)
        except (ValueError, IndexError):
            continue

        for opt_ref_label in ['Opt', 'Ref']:
            mask = folder_df['File'].str.contains(opt_ref_label, na=False)
            subset = folder_df.loc[mask]
            cytotox_vals = subset['Cytotoxicity'].dropna().values

            if len(cytotox_vals) == 0:
                continue

            mean_val = np.mean(cytotox_vals)
            std_val = np.std(cytotox_vals, ddof=1) if len(cytotox_vals) > 1 else 0.0
            cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan

            records.append({
                'folder': folder_name,
                'day': day,
                'condition': condition,
                'opt_ref': opt_ref_label,
                'n_values': len(cytotox_vals),
                'cytotoxicity_mean': mean_val,
                'cytotoxicity_std': std_val,
                'cv_percent': cv,
            })

    return pd.DataFrame(records)


def aggregate_manual_results(manual_df: pd.DataFrame) -> pd.DataFrame:
    if manual_df.empty:
        return pd.DataFrame()

    manual_avg = manual_df.groupby(['day', 'condition', 'opt_ref']).agg(
        manual_mean=('mean_tech', 'mean'),
        manual_std=('mean_tech', 'std'),
        n_manual=('mean_tech', 'count'),
    ).reset_index()

    manual_avg['manual_cv'] = manual_avg.apply(
        lambda row: (row['manual_std'] / row['manual_mean'] * 100)
        if pd.notna(row['manual_mean']) and row['manual_mean'] != 0 else np.nan,
        axis=1,
    )
    return manual_avg


def normalize_analysis_cutoffs(
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    if not apply_cutoffs:
        return None, None

    min_val = None if min_cytotox is None or min_cytotox < 0 else float(min_cytotox)
    max_val = None if max_cv is None or max_cv <= 0 else float(max_cv)
    return min_val, max_val


def add_analysis_cutoff_flags(
    df: pd.DataFrame,
    value_col: str,
    cv_col: str,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    min_val, max_val = normalize_analysis_cutoffs(
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )

    flagged = df.copy()
    pass_min = pd.Series(True, index=flagged.index)
    pass_cv = pd.Series(True, index=flagged.index)

    if min_val is not None and value_col in flagged.columns:
        pass_min = flagged[value_col].notna() & flagged[value_col].ge(min_val)
    if max_val is not None and cv_col in flagged.columns:
        pass_cv = flagged[cv_col].notna() & flagged[cv_col].le(max_val)

    flagged['passes_min_cytotox'] = pass_min
    flagged['passes_max_cv'] = pass_cv
    flagged['passes_analysis_cutoffs'] = pass_min & pass_cv
    return flagged


def filter_analysis_cutoffs(
    df: pd.DataFrame,
    value_col: str,
    cv_col: str,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> pd.DataFrame:
    flagged = add_analysis_cutoff_flags(
        df,
        value_col=value_col,
        cv_col=cv_col,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if 'passes_analysis_cutoffs' not in flagged.columns:
        return flagged
    return flagged.loc[flagged['passes_analysis_cutoffs']].copy()


def compute_comparison_metrics(
    auto_agg: pd.DataFrame,
    manual_df: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    if manual_df.empty or auto_agg.empty:
        return pd.DataFrame(), {'n_comparisons': 0}

    manual_avg = aggregate_manual_results(manual_df)
    auto_analysis = filter_analysis_cutoffs(
        auto_agg,
        value_col='cytotoxicity_mean',
        cv_col='cv_percent',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    manual_analysis = filter_analysis_cutoffs(
        manual_avg,
        value_col='manual_mean',
        cv_col='manual_cv',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )

    comparison = pd.merge(
        auto_analysis[['day', 'condition', 'opt_ref', 'cytotoxicity_mean',
                       'cytotoxicity_std', 'cv_percent', 'n_values']],
        manual_analysis,
        on=['day', 'condition', 'opt_ref'],
        how='inner',
    )

    comparison['diff'] = comparison['cytotoxicity_mean'] - comparison['manual_mean']
    comparison['abs_diff'] = comparison['diff'].abs()
    comparison['pct_diff'] = comparison.apply(
        lambda row: (row['diff'] / row['manual_mean'] * 100)
        if pd.notna(row['manual_mean']) and row['manual_mean'] != 0 else np.nan,
        axis=1,
    )

    valid = comparison.dropna(subset=['cytotoxicity_mean', 'manual_mean'])
    if len(valid) > 2:
        auto_vals = valid['cytotoxicity_mean'].values
        manual_vals = valid['manual_mean'].values
        corr, p_val = stats.pearsonr(auto_vals, manual_vals)
        spearman_r, spearman_p = stats.spearmanr(auto_vals, manual_vals)
        ccc, ccc_precision, ccc_accuracy = concordance_correlation_coefficient(auto_vals, manual_vals)

        diff = auto_vals - manual_vals
        ba_mean_diff = float(diff.mean())
        ba_std_diff = float(diff.std(ddof=1))
        ba_loa_upper = ba_mean_diff + 1.96 * ba_std_diff
        ba_loa_lower = ba_mean_diff - 1.96 * ba_std_diff

        mean_am = (auto_vals + manual_vals) / 2
        slope, _, _, p_prop, _ = stats.linregress(mean_am, diff)

        summary = {
            'n_comparisons': len(valid),
            'pearson_r': corr,
            'pearson_p': p_val,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'ccc': ccc,
            'ccc_precision': ccc_precision,
            'ccc_accuracy': ccc_accuracy,
            'mae': np.mean(np.abs(auto_vals - manual_vals)),
            'rmse': np.sqrt(np.mean((auto_vals - manual_vals) ** 2)),
            'mean_bias': ba_mean_diff,
            'ba_loa_upper': ba_loa_upper,
            'ba_loa_lower': ba_loa_lower,
            'proportional_bias_slope': slope,
            'proportional_bias_p': p_prop,
            'median_abs_diff': np.median(np.abs(auto_vals - manual_vals)),
            'mean_auto_cv': valid['cv_percent'].mean(),
            'mean_manual_cv': valid['manual_cv'].mean(),
        }
    elif len(valid) > 0:
        auto_vals = valid['cytotoxicity_mean'].values
        manual_vals = valid['manual_mean'].values
        summary = {
            'n_comparisons': len(valid),
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'ccc': np.nan,
            'ccc_precision': np.nan,
            'ccc_accuracy': np.nan,
            'mae': np.mean(np.abs(auto_vals - manual_vals)),
            'rmse': np.sqrt(np.mean((auto_vals - manual_vals) ** 2)),
            'mean_bias': np.mean(auto_vals - manual_vals),
            'ba_loa_upper': np.nan,
            'ba_loa_lower': np.nan,
            'proportional_bias_slope': np.nan,
            'proportional_bias_p': np.nan,
            'median_abs_diff': np.median(np.abs(auto_vals - manual_vals)),
            'mean_auto_cv': valid['cv_percent'].mean(),
            'mean_manual_cv': valid['manual_cv'].mean(),
        }
    else:
        summary = {'n_comparisons': 0}

    return comparison, summary


def discover_day_folders(data_path: str) -> List[str]:
    all_folders = get_subfolders(data_path)
    return sorted(folder for folder in all_folders if folder.startswith('Day '))


def select_day_folders(day_folders: List[str], mwe: bool, mwe_n: int, mwe_seed: int) -> List[str]:
    if not mwe:
        return day_folders

    n_folders = min(mwe_n, len(day_folders))
    chooser = random.Random(mwe_seed)
    return sorted(chooser.sample(day_folders, n_folders))


def process_single_folder(
    folder_name: str,
    args: argparse.Namespace,
    channels: ChannelConfig,
    gating: GatingConfig,
    plotting: PlotConfig,
    identifier_handler: IdentifierHandler,
) -> Optional[pd.DataFrame]:
    folder_path = os.path.join(args.data_path, folder_name)
    folder_output = os.path.join(args.output_path, folder_name)

    evaluator = CytotoxEvaluator(
        folder_path,
        identifier_handler,
        channels=channels,
        gating=gating,
        plotting=plotting,
        output_path=folder_output,
    )

    try:
        results = evaluator.run_full_analysis(
            beads_params={},
            cytotox_params={
                'effector_cell': args.effector_cell,
                'target_cell': args.target_cell,
                'proximity_method': args.method,
            },
        )
    except Exception:
        if plotting.enable_all:
            no_plot = PlotConfig(enable_all=False)
            evaluator_retry = CytotoxEvaluator(
                folder_path,
                identifier_handler,
                channels=channels,
                gating=gating,
                plotting=no_plot,
                output_path=folder_output,
            )
            results = evaluator_retry.run_full_analysis(
                beads_params={},
                cytotox_params={
                    'effector_cell': args.effector_cell,
                    'target_cell': args.target_cell,
                    'proximity_method': args.method,
                },
            )
        else:
            raise

    if 'Folder' not in results.columns:
        results.insert(0, 'Folder', folder_name)

    return results


def process_folders(
    folders: List[str],
    args: argparse.Namespace,
    channels: ChannelConfig,
    gating: GatingConfig,
    plotting: PlotConfig,
    identifier_handler: IdentifierHandler,
) -> pd.DataFrame:
    all_results = []
    errors = []

    for index, folder_name in enumerate(folders, 1):
        if args.verbose:
            print(f"  [{index}/{len(folders)}] {folder_name}...", end='', flush=True)

        try:
            results = process_single_folder(folder_name, args, channels, gating, plotting, identifier_handler)

            if results is not None and not results.empty:
                all_results.append(results)
                if args.verbose:
                    n_cytotox = results['Cytotoxicity'].dropna().shape[0]
                    print(f" OK ({len(results)} rows, {n_cytotox} cytotox values)")
            elif args.verbose:
                print(' EMPTY')
        except Exception as exc:
            errors.append((folder_name, str(exc)))
            if args.verbose:
                print(f" ERROR: {exc}")

    if errors and args.verbose:
        print(f"\n  {len(errors)} folder(s) had errors:")
        for folder_name, error in errors:
            print(f"    - {folder_name}: {error}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
