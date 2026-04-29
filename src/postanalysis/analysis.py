#!/usr/bin/env python3
"""
Manuscript-aligned postanalysis for AutoCytotox.

- pooled manual agreement metrics against autogating
- ICC(3,1) for manual raters and manual plus autogating
- manual mixed-effects variance decomposition
- leave-one-out Bland-Altman bias for each evaluator
- residual normality diagnostics for the mixed-effects model

Author: Aleksander Szarzynski TUW 2026
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
from scipy import stats  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

from .. import paths as config
from ..data_loader import create_folder
from ..workflow import (
    DEFAULT_ANALYSIS_MAX_CV,
    DEFAULT_ANALYSIS_MIN_CYTOTOX,
    add_analysis_cutoff_flags,
    aggregate_autogating_results,
    compute_comparison_metrics,
    filter_analysis_cutoffs,
    parse_manual_excel,
)


GROUP_COLS = ['day', 'condition', 'opt_ref']
AUTO_RATER = 'Autogating'
AGREEMENT_METRIC_KEYS = [
    'n_comparisons',
    'pearson_r',
    'pearson_p',
    'spearman_r',
    'spearman_p',
    'ccc',
    'mae',
    'rmse',
    'median_abs_diff',
    'mean_bias',
    'mean_auto_cv',
    'mean_manual_cv',
]


def _manual_analyst_averages(
    manual_df: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> pd.DataFrame:
    """Return one filtered manual value per analyst and experimental group."""
    if manual_df.empty:
        return pd.DataFrame()

    analyst_avgs = manual_df.groupby(
        ['analyst', *GROUP_COLS]
    ).agg(
        cytotox=('mean_tech', 'mean'),
        cytotox_std=('mean_tech', 'std'),
        n_bio_reps=('mean_tech', 'count'),
    ).reset_index()

    analyst_avgs['manual_cv'] = analyst_avgs.apply(
        lambda row: (row['cytotox_std'] / row['cytotox'] * 100)
        if pd.notna(row['cytotox']) and row['cytotox'] != 0 else np.nan,
        axis=1,
    )

    return filter_analysis_cutoffs(
        analyst_avgs,
        value_col='cytotox',
        cv_col='manual_cv',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )


def _filtered_auto_agg(
    auto_agg: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> pd.DataFrame:
    """Return autogating aggregates that pass the manuscript cutoffs."""
    return filter_analysis_cutoffs(
        auto_agg,
        value_col='cytotoxicity_mean',
        cv_col='cv_percent',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )


def matched_rater_table(
    auto_agg: pd.DataFrame,
    manual_df: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> pd.DataFrame:
    """
    Build the complete-case rater table used for ICC and leave-one-out bias.

    Rows contain only groups where all manual users and autogating have a
    retained value after the cytotoxicity and CV cutoffs.
    """
    manual_avgs = _manual_analyst_averages(
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if manual_avgs.empty:
        return pd.DataFrame()

    manual_wide = manual_avgs.pivot_table(
        index=GROUP_COLS,
        columns='analyst',
        values='cytotox',
        aggfunc='mean',
    )
    manual_wide.columns.name = None
    manual_wide = manual_wide.dropna()

    auto_for_merge = _filtered_auto_agg(
        auto_agg,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if auto_for_merge.empty:
        return pd.DataFrame()

    auto_for_merge = auto_for_merge[
        [*GROUP_COLS, 'cytotoxicity_mean']
    ].rename(columns={'cytotoxicity_mean': AUTO_RATER})

    matched = manual_wide.reset_index().merge(
        auto_for_merge,
        on=GROUP_COLS,
        how='inner',
    ).dropna()

    return matched.reset_index(drop=True)


# ============================================================================
# ICC - Intraclass Correlation Coefficient
# ============================================================================

def compute_icc(
    auto_agg: pd.DataFrame,
    manual_df: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> Dict:
    """
    Compute ICC(3,1), two-way mixed, single-measure, consistency.

    The manual-only ICC uses the three manual users. The second ICC adds
    autogating as a fourth fixed rater on the same complete-case groups.
    """
    matched = matched_rater_table(
        auto_agg,
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if matched.empty:
        empty = {'icc': np.nan, 'n': 0, 'k': 0}
        return {
            'icc_manual_only': empty,
            'icc_with_auto': empty,
            'detail_df': pd.DataFrame(),
        }

    manual_cols = [
        col for col in matched.columns
        if col not in GROUP_COLS and col != AUTO_RATER
    ]
    rater_cols = [*manual_cols, AUTO_RATER]

    def _icc_31(ratings_wide: pd.DataFrame) -> dict:
        n = len(ratings_wide)
        k = ratings_wide.shape[1]
        if n < 3 or k < 2:
            return {'icc': np.nan, 'n': n, 'k': k}

        x = ratings_wide.to_numpy(dtype=float)
        grand_mean = x.mean()
        row_means = x.mean(axis=1)
        col_means = x.mean(axis=0)

        ss_rows = k * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_total = np.sum((x - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n - 1)
        ms_error = ss_error / ((n - 1) * (k - 1))

        denom = ms_rows + (k - 1) * ms_error
        icc = (ms_rows - ms_error) / denom if denom > 0 else np.nan
        f_val = ms_rows / ms_error if ms_error > 0 else np.nan
        df1 = n - 1
        df2 = (n - 1) * (k - 1)
        p_val = 1 - stats.f.cdf(f_val, df1, df2) if np.isfinite(f_val) else np.nan

        return {
            'icc': float(icc) if np.isfinite(icc) else np.nan,
            'f': float(f_val) if np.isfinite(f_val) else np.nan,
            'df1': int(df1),
            'df2': int(df2),
            'p': float(p_val) if np.isfinite(p_val) else np.nan,
            'n': int(n),
            'k': int(k),
            'MS_rows': float(ms_rows),
            'MS_error': float(ms_error),
        }

    detail_df = matched[GROUP_COLS + rater_cols].melt(
        id_vars=GROUP_COLS,
        var_name='rater',
        value_name='cytotoxicity',
    )

    return {
        'icc_manual_only': _icc_31(matched[manual_cols]),
        'icc_with_auto': _icc_31(matched[rater_cols]),
        'detail_df': detail_df,
    }


# ============================================================================
# Variance Decomposition - Mixed-Effects Model
# ============================================================================

def variance_decomposition(
    manual_df: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> Dict:
    """
    Decompose manual cytotoxicity variance with a mixed-effects model.

    The full model uses fixed media/culture-mode effects, a day random
    intercept, and a user variance component. It is fitted by REML for
    variance estimates, while ML fits of full and reduced models are compared
    by likelihood-ratio test and AIC.
    """
    import statsmodels.formula.api as smf  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

    analyst_avgs = _manual_analyst_averages(
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if analyst_avgs.empty:
        return {'error': 'No manual user evaluations passed the analysis cutoffs.'}

    analyst_avgs = analyst_avgs.copy()
    analyst_avgs['day_str'] = analyst_avgs['day'].astype(str)
    analyst_avgs['opt_ref_str'] = analyst_avgs['opt_ref'].astype(str)
    analyst_avgs['condition_str'] = analyst_avgs['condition'].astype(str)

    formula = "cytotox ~ C(opt_ref_str) + C(condition_str)"

    try:
        reduced_model = smf.mixedlm(
            formula,
            data=analyst_avgs,
            groups=analyst_avgs['day_str'],
            re_formula='1',
        )
        full_model = smf.mixedlm(
            formula,
            data=analyst_avgs,
            groups=analyst_avgs['day_str'],
            re_formula='1',
            vc_formula={'user': '0 + C(analyst)'},
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reduced_ml = reduced_model.fit(reml=False)
            full_ml = full_model.fit(reml=False)
            result = full_model.fit(reml=True)
    except Exception as exc:
        return {'error': str(exc)}

    residual_var = float(result.scale)
    day_var = float(result.cov_re.iloc[0, 0])
    user_var = float(result.vcomp[0]) if len(result.vcomp) > 0 else 0.0
    total_var = day_var + user_var + residual_var

    components = {
        'Day (random intercept)': day_var,
        'User (manual gating variance component)': user_var,
        'Residual': residual_var,
        'Total': total_var,
    }

    variance_df = pd.DataFrame([
        {
            'source': source,
            'variance': variance,
            'pct_of_total': (variance / total_var * 100)
            if total_var > 0 else np.nan,
        }
        for source, variance in components.items()
    ])

    fit_statistics = {
        'n_obs': int(result.nobs),
        'reduced_model_converged': bool(reduced_ml.converged),
        'full_model_converged': bool(full_ml.converged),
        'reml_full_model_converged': bool(result.converged),
        'reduced_log_likelihood': float(reduced_ml.llf),
        'full_log_likelihood': float(full_ml.llf),
        'reduced_aic': float(reduced_ml.aic),
        'full_aic': float(full_ml.aic),
    }

    lr_stat = 2.0 * (full_ml.llf - reduced_ml.llf)
    lr_df = max(len(full_ml.params) - len(reduced_ml.params), 1)
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    model_comparison = {
        'comparison': 'Full model vs reduced model without user variance',
        'likelihood_ratio_statistic': float(lr_stat),
        'df_difference': int(lr_df),
        'p_value': float(lr_p),
        'supports_user_component': bool(lr_p < 0.05),
    }

    fixed_effects_df = pd.DataFrame({
        'term': result.fe_params.index,
        'coefficient': result.fe_params.values,
        'std_error': result.bse_fe.values,
        'z_value': result.fe_params.values / result.bse_fe.values,
        'p_value': result.pvalues[result.fe_params.index].values,
    })

    diagnostics_detail_df = analyst_avgs[
        ['analyst', 'day', 'condition', 'opt_ref', 'cytotox']
    ].copy()
    diagnostics_detail_df['fitted_value'] = result.fittedvalues
    diagnostics_detail_df['residual'] = result.resid
    diagnostics_detail_df['standardized_residual'] = (
        diagnostics_detail_df['residual'] / np.sqrt(residual_var)
        if residual_var > 0 else np.nan
    )

    if len(diagnostics_detail_df) >= 3:
        shapiro_w, shapiro_p = stats.shapiro(diagnostics_detail_df['residual'])
    else:
        shapiro_w, shapiro_p = np.nan, np.nan

    diagnostics_summary_df = pd.DataFrame([
        {
            'metric': 'n_residuals',
            'value': int(len(diagnostics_detail_df)),
            'interpretation': 'Number of residuals used for diagnostics.',
        },
        {
            'metric': 'shapiro_wilk_W',
            'value': float(shapiro_w) if np.isfinite(shapiro_w) else np.nan,
            'interpretation': 'Higher values indicate closer agreement with normality.',
        },
        {
            'metric': 'shapiro_wilk_p',
            'value': float(shapiro_p) if np.isfinite(shapiro_p) else np.nan,
            'interpretation': 'Large p-values support approximate residual normality.',
        },
    ])

    model_fit_df = pd.DataFrame([
        {'metric': key, 'value': value}
        for key, value in fit_statistics.items()
    ])
    model_comparison_df = pd.DataFrame([
        {'metric': key, 'value': value}
        for key, value in model_comparison.items()
    ])

    user_pct = (user_var / total_var * 100) if total_var > 0 else np.nan

    return {
        'manual_model_summary': str(result.summary()),
        'reduced_model_summary': str(reduced_ml.summary()),
        'fit_statistics': fit_statistics,
        'model_comparison': model_comparison,
        'model_fit_df': model_fit_df,
        'model_comparison_df': model_comparison_df,
        'fixed_effects_df': fixed_effects_df,
        'diagnostics_summary_df': diagnostics_summary_df,
        'diagnostics_detail_df': diagnostics_detail_df,
        'variance_components': components,
        'variance_components_df': variance_df,
        'user_pct': user_pct,
        'model_data': analyst_avgs,
    }


# ============================================================================
# Leave-One-Out Bland-Altman
# ============================================================================

def leave_one_out_bland_altman(
    auto_agg: pd.DataFrame,
    manual_df: pd.DataFrame,
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute leave-one-out Bland-Altman bias for each evaluator.

    Each manual user and autogating are compared with the pooled mean of the
    other three evaluators on complete-case groups.
    """
    matched = matched_rater_table(
        auto_agg,
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    rater_cols = [col for col in matched.columns if col not in GROUP_COLS]
    records = []

    for evaluator in rater_cols:
        peers = [col for col in rater_cols if col != evaluator]
        reference = matched[peers].mean(axis=1)
        evaluator_values = matched[evaluator]
        diff = evaluator_values - reference
        ba_mean = (evaluator_values + reference) / 2

        for idx, row in matched.iterrows():
            records.append({
                'evaluator': evaluator,
                'day': row['day'],
                'condition': row['condition'],
                'opt_ref': row['opt_ref'],
                'evaluator_cytotoxicity': evaluator_values.loc[idx],
                'pooled_reference': reference.loc[idx],
                'ba_mean': ba_mean.loc[idx],
                'ba_diff': diff.loc[idx],
            })

    detail_df = pd.DataFrame(records)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    summary_df = detail_df.groupby('evaluator').agg(
        n=('ba_diff', 'count'),
        mean_bias=('ba_diff', 'mean'),
        sd_diff=('ba_diff', 'std'),
        median_abs_diff=('ba_diff', lambda s: float(np.median(np.abs(s)))),
    ).reset_index()
    summary_df['loa_lower'] = summary_df['mean_bias'] - 1.96 * summary_df['sd_diff']
    summary_df['loa_upper'] = summary_df['mean_bias'] + 1.96 * summary_df['sd_diff']

    return detail_df, summary_df


# ============================================================================
# Plotting
# ============================================================================

def plot_leave_one_out_bland_altman(
    ba_df: pd.DataFrame,
    output_path: str,
    fmt: str = 'png',
) -> None:
    """Plot leave-one-out Bland-Altman panels for all evaluators."""
    if ba_df.empty:
        return

    evaluators = sorted(ba_df['evaluator'].unique())
    n = len(evaluators)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for ax, evaluator in zip(axes.flatten(), evaluators):
        sub = ba_df[ba_df['evaluator'] == evaluator]
        ax.scatter(
            sub['ba_mean'],
            sub['ba_diff'],
            alpha=0.65,
            s=38,
            edgecolors='white',
            linewidth=0.5,
        )
        mean_diff = sub['ba_diff'].mean()
        sd_diff = sub['ba_diff'].std()
        loa_upper = mean_diff + 1.96 * sd_diff
        loa_lower = mean_diff - 1.96 * sd_diff

        ax.axhline(mean_diff, color='black', linewidth=1.2,
                   label=f'Bias: {mean_diff:+.2f}')
        ax.axhline(loa_upper, color='red', linestyle='--', alpha=0.7,
                   label=f'+1.96 SD: {loa_upper:+.2f}')
        ax.axhline(loa_lower, color='red', linestyle='--', alpha=0.7,
                   label=f'-1.96 SD: {loa_lower:+.2f}')
        ax.axhline(0, color='gray', linewidth=0.7)
        ax.set_xlabel('Mean of evaluator and pooled reference (%)')
        ax.set_ylabel('Evaluator - pooled reference (%)')
        ax.set_title(f'{evaluator} (n={len(sub)})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    for ax in axes.flatten()[len(evaluators):]:
        ax.axis('off')

    plt.suptitle('Leave-One-Out Bland-Altman Bias', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f'bland_altman_leave_one_out.{fmt}'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()


def plot_icc_comparison(icc_results: Dict, output_path: str, fmt: str = 'png') -> None:
    """Bar chart comparing ICC(3,1) with and without autogating."""
    icc_manual = icc_results.get('icc_manual_only', {})
    icc_auto = icc_results.get('icc_with_auto', {})
    if np.isnan(icc_manual.get('icc', np.nan)) or np.isnan(icc_auto.get('icc', np.nan)):
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Manual only\n(3 users)', 'Manual + autogating\n(4 raters)']
    values = [icc_manual['icc'], icc_auto['icc']]
    bars = ax.bar(labels, values, color=['#377eb8', '#ff7f00'],
                  edgecolor='white', linewidth=1.5, width=0.55)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )

    ax.set_ylabel('ICC(3,1)')
    ax.set_title('Rater Consistency')
    ax.set_ylim(0, min(1.05, max(values) + 0.1))
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'icc_comparison.{fmt}'), dpi=150)
    plt.close()


def plot_variance_decomposition(var_results: Dict, output_path: str, fmt: str = 'png') -> None:
    """Plot mixed-model variance components."""
    components = var_results.get('variance_components')
    if not components:
        return

    labels = [label for label in components if label != 'Total']
    values = [components[label] for label in labels]
    total = sum(values)
    if total <= 0:
        return

    pcts = [value / total * 100 for value in values]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, pcts, color=['#377eb8', '#ff7f00', '#7f7f7f'])

    for bar, pct, value in zip(bars, pcts, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}% (sigma^2={value:.2f})',
            va='center',
            fontsize=9,
        )

    ax.set_xlabel('% of total variance')
    ax.set_title('Manual Gating Variance Decomposition')
    ax.set_xlim(0, max(pcts) * 1.25 if pcts else 1)
    ax.grid(True, axis='x', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'variance_decomposition.{fmt}'), dpi=150)
    plt.close()


def plot_mixed_model_diagnostics(var_results: Dict, output_path: str, fmt: str = 'png') -> None:
    """Plot residual distribution and Q-Q diagnostics for the mixed model."""
    diag_df = var_results.get('diagnostics_detail_df')
    if diag_df is None or diag_df.empty:
        return

    diag_summary = var_results.get('diagnostics_summary_df', pd.DataFrame())
    shapiro_text = ''
    if not diag_summary.empty:
        diag_map = diag_summary.set_index('metric')['value']
        shapiro_text = (
            f"Shapiro W={diag_map.get('shapiro_wilk_W', np.nan):.3f}, "
            f"p={diag_map.get('shapiro_wilk_p', np.nan):.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        diag_df['residual'],
        bins=min(15, max(8, len(diag_df) // 8)),
        color='#377eb8',
        edgecolor='white',
        alpha=0.85,
    )
    axes[0].axvline(0, linestyle='--', color='gray', linewidth=1)
    axes[0].set_xlabel('Residual (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Residual Distribution')

    qq = stats.probplot(diag_df['residual'], dist='norm')
    theoretical = qq[0][0]
    ordered = qq[0][1]
    slope = float(qq[1][0])
    intercept = float(qq[1][1])
    axes[1].scatter(theoretical, ordered, alpha=0.7, s=35,
                    edgecolors='white', linewidth=0.4)
    axes[1].plot(theoretical, slope * theoretical + intercept,
                 linestyle='--', color='gray', linewidth=1)
    axes[1].set_xlabel('Theoretical quantiles')
    axes[1].set_ylabel('Observed residual quantiles')
    axes[1].set_title('Normal Q-Q Plot')

    title = 'Mixed-Model Residual Diagnostics'
    if shapiro_text:
        title = f'{title}\n{shapiro_text}'
    plt.suptitle(title, fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f'mixed_model_diagnostics.{fmt}'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()


# ============================================================================
# Output
# ============================================================================

def export_csvs(
    output_path: str,
    auto_agg: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame],
    agreement_metrics_df: Optional[pd.DataFrame],
    ba_detail_df: Optional[pd.DataFrame],
    ba_summary_df: Optional[pd.DataFrame],
    icc_detail_df: Optional[pd.DataFrame],
    icc_summary_df: Optional[pd.DataFrame],
    variance_df: Optional[pd.DataFrame],
    model_fit_df: Optional[pd.DataFrame],
    model_comparison_df: Optional[pd.DataFrame],
    fixed_effects_df: Optional[pd.DataFrame],
    diagnostics_summary_df: Optional[pd.DataFrame],
    diagnostics_detail_df: Optional[pd.DataFrame],
) -> Tuple[str, List[str]]:
    """Export retained postanalysis tables as CSV files."""
    csv_dir = os.path.join(output_path, 'csv_export')
    create_folder(csv_dir)

    exports = {
        'autogating_aggregated.csv': auto_agg,
        'agreement_comparison.csv': comparison_df,
        'agreement_metrics.csv': agreement_metrics_df,
        'bland_altman_leave_one_out_detail.csv': ba_detail_df,
        'bland_altman_leave_one_out_summary.csv': ba_summary_df,
        'icc_rater_detail.csv': icc_detail_df,
        'icc_summary.csv': icc_summary_df,
        'variance_decomposition.csv': variance_df,
        'model_fit.csv': model_fit_df,
        'model_comparison.csv': model_comparison_df,
        'fixed_effects.csv': fixed_effects_df,
        'mixed_model_diagnostics_summary.csv': diagnostics_summary_df,
        'mixed_model_diagnostics_detail.csv': diagnostics_detail_df,
    }

    saved = []
    for filename, df in exports.items():
        if df is not None and not df.empty:
            path = os.path.join(csv_dir, filename)
            df.to_csv(path, index=False)
            saved.append(filename)

    return csv_dir, saved


def save_analysis_excel(
    output_path: str,
    auto_agg: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame],
    agreement_metrics_df: Optional[pd.DataFrame],
    ba_detail_df: Optional[pd.DataFrame],
    ba_summary_df: Optional[pd.DataFrame],
    icc_detail_df: Optional[pd.DataFrame],
    icc_summary_df: Optional[pd.DataFrame],
    variance_df: Optional[pd.DataFrame],
    model_fit_df: Optional[pd.DataFrame],
    model_comparison_df: Optional[pd.DataFrame],
    fixed_effects_df: Optional[pd.DataFrame],
    diagnostics_summary_df: Optional[pd.DataFrame],
    diagnostics_detail_df: Optional[pd.DataFrame],
) -> str:
    """Save retained postanalysis tables to a multi-sheet Excel workbook."""
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    filepath = os.path.join(output_path, f'postanalysis_{timestamp}.xlsx')

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        auto_agg.to_excel(writer, sheet_name='Aggregated', index=False)

        optional_sheets = [
            ('Agreement', comparison_df),
            ('Agreement_Metrics', agreement_metrics_df),
            ('Bland_Altman_LOO', ba_detail_df),
            ('Bland_Altman_Summary', ba_summary_df),
            ('ICC_Detail', icc_detail_df),
            ('ICC_Summary', icc_summary_df),
            ('Variance_Decomposition', variance_df),
            ('Model_Fit', model_fit_df),
            ('Model_Comparison', model_comparison_df),
            ('Fixed_Effects', fixed_effects_df),
            ('Model_Diagnostics', diagnostics_summary_df),
            ('Diagnostics_Detail', diagnostics_detail_df),
        ]
        for sheet_name, df in optional_sheets:
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    return filepath


# ============================================================================
# Main analysis
# ============================================================================

def run_analysis(
    results_excel: str,
    manual_excel: str,
    output_path: str,
    export_csv: bool = False,
    plot_format: str = 'png',
    min_cytotox: Optional[float] = DEFAULT_ANALYSIS_MIN_CYTOTOX,
    max_cv: Optional[float] = DEFAULT_ANALYSIS_MAX_CV,
    apply_cutoffs: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run the manuscript-aligned postanalysis workflow."""
    if verbose:
        print(f'\nLoading results from: {results_excel}')
    combined_results = pd.read_excel(results_excel, sheet_name='Raw_Results')
    auto_agg = add_analysis_cutoff_flags(
        aggregate_autogating_results(combined_results),
        value_col='cytotoxicity_mean',
        cv_col='cv_percent',
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    analysis_auto_agg = _filtered_auto_agg(
        auto_agg,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )

    if verbose:
        print(f'  {len(combined_results)} raw rows, {len(auto_agg)} aggregated groups')
        if apply_cutoffs and 'passes_analysis_cutoffs' in auto_agg.columns:
            retained = int(auto_agg['passes_analysis_cutoffs'].sum())
            print(f'  Analysis groups retained after cutoffs: {retained} / {len(auto_agg)}')

    if verbose:
        print(f'Loading manual data from: {manual_excel}')
    manual_df = parse_manual_excel(manual_excel)
    if verbose:
        print(f"  {len(manual_df)} manual records ({manual_df['analyst'].nunique()} users)")

    comparison_df, agreement_metrics = compute_comparison_metrics(
        auto_agg,
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    agreement_metrics = {
        key: agreement_metrics.get(key, np.nan)
        for key in AGREEMENT_METRIC_KEYS
    }
    agreement_metrics_df = pd.DataFrame([agreement_metrics])

    if verbose:
        print('\nPooled manual agreement metrics...')
        print(f"  n={agreement_metrics.get('n_comparisons', 0)}")
        for key in ['pearson_r', 'spearman_r', 'ccc', 'mae', 'rmse',
                    'median_abs_diff', 'mean_bias']:
            value = agreement_metrics.get(key, np.nan)
            if pd.notna(value):
                print(f'  {key}: {value:.4f}')

    if verbose:
        print('\nICC analysis...')
    icc_results = compute_icc(
        auto_agg,
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    icc_detail_df = icc_results.get('detail_df', pd.DataFrame())
    icc_summary_records = []
    for scope, key in [
        ('Manual only (3 users)', 'icc_manual_only'),
        ('Manual + autogating (4 raters)', 'icc_with_auto'),
    ]:
        row = icc_results.get(key, {})
        icc_summary_records.append({
            'scope': scope,
            'icc_31': row.get('icc', np.nan),
            'F': row.get('f', np.nan),
            'df1': row.get('df1', np.nan),
            'df2': row.get('df2', np.nan),
            'p': row.get('p', np.nan),
            'n_targets': row.get('n', np.nan),
            'k_raters': row.get('k', np.nan),
        })
    icc_summary_df = pd.DataFrame(icc_summary_records)

    if verbose:
        for _, row in icc_summary_df.iterrows():
            print(f"  {row['scope']}: ICC(3,1)={row['icc_31']:.3f}, "
                  f"n={row['n_targets']:.0f}, p={row['p']:.4g}")

    if verbose:
        print('\nVariance decomposition (manual mixed-effects model)...')
    var_results = variance_decomposition(
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )

    variance_df = pd.DataFrame()
    model_fit_df = pd.DataFrame()
    model_comparison_df = pd.DataFrame()
    fixed_effects_df = pd.DataFrame()
    diagnostics_summary_df = pd.DataFrame()
    diagnostics_detail_df = pd.DataFrame()

    if 'error' not in var_results:
        variance_df = var_results.get('variance_components_df', pd.DataFrame())
        model_fit_df = var_results.get('model_fit_df', pd.DataFrame())
        model_comparison_df = var_results.get('model_comparison_df', pd.DataFrame())
        fixed_effects_df = var_results.get('fixed_effects_df', pd.DataFrame())
        diagnostics_summary_df = var_results.get('diagnostics_summary_df', pd.DataFrame())
        diagnostics_detail_df = var_results.get('diagnostics_detail_df', pd.DataFrame())

        if verbose:
            components = var_results['variance_components']
            print(f"  User variance: {var_results['user_pct']:.2f}% of total")
            for source, variance in components.items():
                if source == 'Total':
                    continue
                pct = variance / components['Total'] * 100 if components['Total'] > 0 else np.nan
                print(f'    {source}: sigma^2={variance:.2f} ({pct:.2f}%)')
            fit_stats = var_results.get('fit_statistics', {})
            comparison = var_results.get('model_comparison', {})
            print(
                f"  ML comparison: full LL={fit_stats.get('full_log_likelihood', np.nan):.2f}, "
                f"reduced LL={fit_stats.get('reduced_log_likelihood', np.nan):.2f}, "
                f"full AIC={fit_stats.get('full_aic', np.nan):.2f}, "
                f"reduced AIC={fit_stats.get('reduced_aic', np.nan):.2f}"
            )
            print(
                f"  LRT: chi2={comparison.get('likelihood_ratio_statistic', np.nan):.2f}, "
                f"p={comparison.get('p_value', np.nan):.4g}"
            )
            if not diagnostics_summary_df.empty:
                diag_map = diagnostics_summary_df.set_index('metric')['value']
                print(
                    f"  Shapiro-Wilk: W={diag_map.get('shapiro_wilk_W', np.nan):.3f}, "
                    f"p={diag_map.get('shapiro_wilk_p', np.nan):.4g}"
                )
    elif verbose:
        print(f"  WARNING: mixed model failed: {var_results['error']}")

    if verbose:
        print('\nLeave-one-out Bland-Altman bias...')
    ba_df, ba_summary_df = leave_one_out_bland_altman(
        auto_agg,
        manual_df,
        min_cytotox=min_cytotox,
        max_cv=max_cv,
        apply_cutoffs=apply_cutoffs,
    )
    if verbose and not ba_summary_df.empty:
        for _, row in ba_summary_df.iterrows():
            print(
                f"  {row['evaluator']}: bias={row['mean_bias']:+.2f}%, "
                f"LoA=[{row['loa_lower']:+.2f}, {row['loa_upper']:+.2f}]%"
            )

    plots_path = os.path.join(output_path, 'analysis_plots')
    create_folder(plots_path)
    if not ba_df.empty:
        plot_leave_one_out_bland_altman(ba_df, plots_path, plot_format)
    if 'icc_manual_only' in icc_results:
        plot_icc_comparison(icc_results, plots_path, plot_format)
    if 'error' not in var_results:
        plot_variance_decomposition(var_results, plots_path, plot_format)
        plot_mixed_model_diagnostics(var_results, plots_path, plot_format)

    if verbose:
        print(f'\nPlots saved to: {plots_path}')

    csv_dir = None
    if export_csv:
        if verbose:
            print('\nExporting CSVs...')
        csv_dir, saved = export_csvs(
            output_path,
            auto_agg,
            comparison_df,
            agreement_metrics_df,
            ba_df,
            ba_summary_df,
            icc_detail_df,
            icc_summary_df,
            variance_df,
            model_fit_df,
            model_comparison_df,
            fixed_effects_df,
            diagnostics_summary_df,
            diagnostics_detail_df,
        )
        if verbose:
            print(f'  {len(saved)} CSVs saved to: {csv_dir}')

    if verbose:
        print('\nSaving analysis Excel...')
    excel_path = save_analysis_excel(
        output_path,
        auto_agg,
        comparison_df,
        agreement_metrics_df,
        ba_df,
        ba_summary_df,
        icc_detail_df,
        icc_summary_df,
        variance_df,
        model_fit_df,
        model_comparison_df,
        fixed_effects_df,
        diagnostics_summary_df,
        diagnostics_detail_df,
    )
    if verbose:
        print(f'  Saved to: {excel_path}')

    return {
        'auto_agg': auto_agg,
        'analysis_auto_agg': analysis_auto_agg,
        'comparison_df': comparison_df,
        'agreement_metrics': agreement_metrics,
        'agreement_metrics_df': agreement_metrics_df,
        'icc_results': icc_results,
        'icc_summary_df': icc_summary_df,
        'variance_results': var_results,
        'variance_df': variance_df,
        'model_fit_df': model_fit_df,
        'model_comparison_df': model_comparison_df,
        'fixed_effects_df': fixed_effects_df,
        'diagnostics_summary_df': diagnostics_summary_df,
        'diagnostics_detail_df': diagnostics_detail_df,
        'bland_altman_df': ba_df,
        'bland_altman_summary_df': ba_summary_df,
        'excel_path': excel_path,
        'csv_dir': csv_dir,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='AutoCytotox manuscript-aligned postanalysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-excel', '--autocytotox-excel',
        dest='results_excel',
        type=str,
        default=None,
        help='Path to the AutoCytotox results Excel file with a Raw_Results sheet.',
    )
    parser.add_argument(
        '--gridsearch-dir',
        type=str,
        default=None,
        help='Path to gridsearch output dir; runs postanalysis on each completed config.',
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
        help='Output directory. Defaults to the directory containing results-excel.',
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Also export retained postanalysis tables as CSV files.',
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
        help='Minimum cytotoxicity percent required before postanalysis. '
             'Set to a negative value to disable.',
    )
    parser.add_argument(
        '--max-analysis-cv',
        type=float,
        default=DEFAULT_ANALYSIS_MAX_CV,
        help='Maximum CV percent allowed before postanalysis. '
             'Set to 0 or a negative value to disable.',
    )
    parser.add_argument(
        '--disable-analysis-cutoffs',
        action='store_true',
        help='Disable the pre-analysis CV and cytotoxicity cutoffs entirely.',
    )
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args(argv)

    if args.results_excel is None and args.gridsearch_dir is None:
        parser.error('Either --results-excel or --gridsearch-dir is required')

    if args.output_path is None and args.results_excel:
        args.output_path = os.path.dirname(os.path.abspath(args.results_excel))

    args.apply_cutoffs = not args.disable_analysis_cutoffs
    return args


def _find_gridsearch_excels(gs_dir: str) -> List[str]:
    """Find completed AutoCytotox result workbooks in gridsearch folders."""
    excels: List[str] = []
    search_roots = [gs_dir]
    legacy_configs_dir = os.path.join(gs_dir, 'configs')
    if os.path.isdir(legacy_configs_dir):
        search_roots.insert(0, legacy_configs_dir)

    seen = set()
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            cfg_path = os.path.join(root, entry)
            if not os.path.isdir(cfg_path):
                continue
            for filename in os.listdir(cfg_path):
                if filename.startswith('autocytotox_results_') and filename.endswith('.xlsx'):
                    excel_path = os.path.join(cfg_path, filename)
                    if excel_path not in seen:
                        excels.append(excel_path)
                        seen.add(excel_path)
                    break
    return excels


def main(argv=None):
    args = parse_args(argv)

    print('=' * 60)
    print('AutoCytotox Postanalysis')
    print('=' * 60)

    if args.gridsearch_dir:
        excels = _find_gridsearch_excels(args.gridsearch_dir)
        if not excels:
            print(f'No autocytotox_results_*.xlsx found under: {args.gridsearch_dir}')
            sys.exit(1)

        out_base = args.output_path or os.path.join(args.gridsearch_dir, 'analysis')
        create_folder(out_base)
        print(f'\nFound {len(excels)} config results to analyse')

        for index, excel in enumerate(excels, 1):
            cfg_name = os.path.basename(os.path.dirname(excel))
            cfg_out = os.path.join(out_base, cfg_name)
            create_folder(cfg_out)
            print(f'\n[{index}/{len(excels)}] {cfg_name}')
            try:
                run_analysis(
                    results_excel=excel,
                    manual_excel=args.manual_excel,
                    output_path=cfg_out,
                    export_csv=args.export_csv,
                    plot_format=args.plot_format,
                    min_cytotox=args.min_analysis_cytotox,
                    max_cv=args.max_analysis_cv,
                    apply_cutoffs=args.apply_cutoffs,
                    verbose=not args.quiet,
                )
            except Exception as exc:
                print(f'  ERROR: {exc}')
    else:
        run_analysis(
            results_excel=args.results_excel,
            manual_excel=args.manual_excel,
            output_path=args.output_path,
            export_csv=args.export_csv,
            plot_format=args.plot_format,
            min_cytotox=args.min_analysis_cytotox,
            max_cv=args.max_analysis_cv,
            apply_cutoffs=args.apply_cutoffs,
            verbose=not args.quiet,
        )

    print(f'\n{"=" * 60}')
    print('POSTANALYSIS COMPLETE')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
