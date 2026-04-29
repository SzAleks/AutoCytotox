"""
End-to-End Pipeline Orchestrator for the autocytotox Autogating Pipeline.

This module ties the loader, gating engine, aggregation, comparison and
reporting layers into a single ``run_from_config`` entrypoint that consumes a
validated runner configuration and produces the results Excel and optional
diagnostic plots.

Author: Aleksander Szarzynski TUW 2026
"""

import os
import time
import warnings
from typing import Any, Dict, Optional

from .. import paths as repo_paths
from ..config_loader import build_args_namespace, build_runtime_configs
from ..data_loader import create_folder
from ..identifiers import resolve_identifier_handler
from ..reporting import render_comparison_plots, save_comprehensive_excel
from .core import (
    add_analysis_cutoff_flags,
    aggregate_autogating_results,
    compute_comparison_metrics,
    discover_day_folders,
    parse_manual_excel,
    process_folders,
    select_day_folders,
)


def _print_header(args) -> None:
    print('=' * 60)
    print('autocytotox - YAML-Driven Autogating Cytotoxicity Analysis')
    print('=' * 60)
    print(f'  Config Path:      {args.config_path}')
    print(f'  Method:           {args.method}')
    print(f'  Percent:          {args.percent}')
    print(f'  BW Adjust:        {args.bw_adjust}')
    print(f'  Min Distance:     {args.min_distance}')
    print(f'  Gate 2 Method:    {args.gate2_method}')
    print(f'  Gate Derivation:  {args.gate_derivation}')
    print(f'  Identifier Mode:  {args.identifier_strategy}')
    if args.identifier_callable:
        print(f'  Identifier Call:  {args.identifier_callable}')
    if args.gate_derivation == 'percentile':
        print(f'  Percentile Q:     {args.percentile_q}')
    elif args.gate_derivation == 'mean_ksd':
        print(f'  K-Sigma:          {args.k_sigma}')
    beads_value = args.beads_x_cutoff
    if beads_value is None:
        beads_label = 'auto (Size Beads file) -- OPT-IN, may drift on noisy beads, use at own risk'
    else:
        beads_label = f'{beads_value} (hard-coded)'
    print(f'  Beads Cutoff:     {beads_label}')
    print(f'  MWE Mode:         {args.mwe} (n={args.mwe_n})')
    if getattr(args, 'subsample', 0) > 0:
        print(f'  Subsample:        {args.subsample * 100:.0f}%')
    print(f'  Plots:            {not args.no_plots}')
    print(f'  Data Path:        {args.data_path}')
    print(f'  Output Path:      {args.output_path}')
    if args.manual_excel:
        print(f'  Manual Excel:     {args.manual_excel}')
    print('=' * 60)


def run_from_config(config_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    args = build_args_namespace(config_data, config_path=config_path)
    channels, gating, plotting = build_runtime_configs(config_data)
    identifier_handler = resolve_identifier_handler(
        strategy=args.identifier_strategy,
        callable_path=args.identifier_callable,
        repo_root=repo_paths.PATH_RAW_DATA,
    )

    create_folder(args.output_path)
    if args.verbose:
        _print_header(args)

    start_time = time.time()
    day_folders = discover_day_folders(args.data_path)
    if args.verbose:
        print(f'\nFound {len(day_folders)} Day folders')

    if not day_folders:
        raise FileNotFoundError(f'No Day folders found under {args.data_path}')

    folders = select_day_folders(day_folders, args.mwe, args.mwe_n, args.mwe_seed)
    if args.mwe and args.verbose:
        print(f'MWE mode: selected {len(folders)} folders: {folders}')

    if args.verbose:
        print(f'\nProcessing {len(folders)} folders...')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        combined_results = process_folders(folders, args, channels, gating, plotting, identifier_handler)

    if combined_results.empty:
        raise RuntimeError('No results generated. Check the configuration and input data.')

    if args.verbose:
        total_rows = len(combined_results)
        n_cytotox = combined_results['Cytotoxicity'].dropna().shape[0]
        print(f'\nTotal: {total_rows} rows, {n_cytotox} cytotoxicity values')
        print('\nAggregating results...')

    auto_agg = add_analysis_cutoff_flags(
        aggregate_autogating_results(combined_results),
        value_col='cytotoxicity_mean',
        cv_col='cv_percent',
        min_cytotox=args.min_analysis_cytotox,
        max_cv=args.max_analysis_cv,
        apply_cutoffs=args.apply_analysis_cutoffs,
    )
    analysis_auto_agg = auto_agg.loc[
        auto_agg['passes_analysis_cutoffs']
    ].copy() if 'passes_analysis_cutoffs' in auto_agg.columns else auto_agg.copy()

    if args.verbose and not auto_agg.empty:
        print(f'  {len(auto_agg)} aggregate groups')
        raw_mean_cv = auto_agg['cv_percent'].dropna().mean()
        print(f'  Raw mean CV%: {raw_mean_cv:.2f}')
        if args.apply_analysis_cutoffs:
            filtered_mean_cv = analysis_auto_agg['cv_percent'].dropna().mean()
            print(f'  Analysis groups retained: {len(analysis_auto_agg)} / {len(auto_agg)}')
            print(f'  Filtered mean CV%: {filtered_mean_cv:.2f}')

    comparison_df = None
    summary_metrics = None
    manual_df = None
    plots_path = None

    if args.run_analysis and not args.manual_excel:
        raise ValueError('--run-analysis requires --manual-excel or paths.manual_excel.')

    if args.manual_excel:
        if not os.path.exists(args.manual_excel):
            raise FileNotFoundError(f'Manual Excel not found: {args.manual_excel}')

        if args.verbose:
            print(f'\nLoading manual analysis from: {args.manual_excel}')

        manual_df = parse_manual_excel(args.manual_excel)

        if args.verbose:
            print(f"  Parsed {len(manual_df)} manual records ({manual_df['analyst'].nunique()} analysts)")

        comparison_df, summary_metrics = compute_comparison_metrics(
            auto_agg,
            manual_df,
            min_cytotox=args.min_analysis_cytotox,
            max_cv=args.max_analysis_cv,
            apply_cutoffs=args.apply_analysis_cutoffs,
        )

        if args.verbose and summary_metrics.get('n_comparisons', 0) > 0:
            print('\n  Comparison Metrics:')
            print(f"  {'-' * 40}")
            for key in [
                'n_comparisons', 'pearson_r', 'spearman_r', 'ccc', 'ccc_precision', 'ccc_accuracy',
                'mae', 'rmse', 'mean_bias', 'ba_loa_upper', 'ba_loa_lower',
                'proportional_bias_slope', 'proportional_bias_p', 'mean_auto_cv', 'mean_manual_cv',
            ]:
                value = summary_metrics.get(key)
                if isinstance(value, float):
                    print(f'  {key:28s}: {value:.4f}')
                else:
                    print(f'  {key:28s}: {value}')

        if not args.no_plots and comparison_df is not None and not comparison_df.empty:
            if args.verbose:
                print('\nGenerating comparison plots...')
            plots_path = render_comparison_plots(
                comparison_df,
                summary_metrics,
                args.output_path,
                args.plot_format,
            )
            if args.verbose:
                print(f'  Plots saved to: {plots_path}')

    if args.verbose:
        print('\nSaving Excel output...')

    excel_path = save_comprehensive_excel(
        args.output_path,
        combined_results,
        auto_agg,
        comparison_df,
        summary_metrics,
        manual_df,
        args,
        prefix='autocytotox',
    )

    analysis_results = None
    if args.run_analysis and manual_df is not None:
        if args.verbose:
            print('\n' + '=' * 60)
            print('Extended Analysis (postanalysis)')
            print('=' * 60)

        from ..postanalysis import run_analysis as run_postanalysis

        analysis_results = run_postanalysis(
            results_excel=excel_path,
            manual_excel=args.manual_excel,
            output_path=args.output_path,
            export_csv=args.export_csv,
            plot_format=args.plot_format,
            min_cytotox=args.min_analysis_cytotox,
            max_cv=args.max_analysis_cv,
            apply_cutoffs=args.apply_analysis_cutoffs,
            verbose=args.verbose,
        )

    elapsed = time.time() - start_time
    if args.verbose:
        print(f"\n{'=' * 60}")
        print(f'COMPLETE - Elapsed: {elapsed:.1f}s')
        print(f'Excel:  {excel_path}')
        print(f'Output: {args.output_path}')
        print(f"{'=' * 60}")

    return {
        'combined_results': combined_results,
        'auto_agg': auto_agg,
        'analysis_auto_agg': analysis_auto_agg,
        'comparison_df': comparison_df,
        'summary_metrics': summary_metrics,
        'manual_df': manual_df,
        'excel_path': excel_path,
        'plots_path': plots_path,
        'analysis_results': analysis_results,
        'args': args,
        'config': config_data,
        'identifier_handler': identifier_handler,
        'selected_folders': folders,
    }
