"""
YAML Configuration Loader for the autocytotox Autogating Pipeline.

This module loads the runner YAML, validates parameter values, applies CLI
overrides, and converts the merged dictionary into the dataclass
configurations (``ChannelConfig``, ``GatingConfig``, ``PlotConfig``) consumed
by the gating engine.

Author: Aleksander Szarzynski TUW 2026
"""

import argparse
import os
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError as exc:
    raise RuntimeError('PyYAML is required for the autocytotox configuration workflow.') from exc

from . import paths as repo_paths
from .cytotox_evaluator import ChannelConfig, GatingConfig, PlotConfig
from .identifiers import get_builtin_identifier_strategy_names
from .workflow import DEFAULT_ANALYSIS_MAX_CV, DEFAULT_ANALYSIS_MIN_CYTOTOX


VALID_METHODS = {'axis', 'euclidean', 'density'}
VALID_GATE2_METHODS = {'knn', 'ratio', 'linear_band'}
VALID_GATE_DERIVATIONS = {'max', 'percentile', 'mean_ksd'}
VALID_PLOT_FORMATS = {'png', 'pdf', 'svg'}
PLOT_TOGGLE_KEYS = [
    'plot_beads',
    'plot_gate_lines',
    'plot_proximity',
    'plot_kde_minima',
    'plot_sc_verification',
    'plot_2sd',
    'plot_quadrant',
]


def _default_plotting_config() -> Dict[str, Any]:
    plotting = asdict(PlotConfig())
    plotting['enable_all'] = False
    plotting['plot_beads'] = False
    plotting['plot_gate_lines'] = False
    plotting['plot_proximity'] = False
    plotting['plot_kde_minima'] = False
    plotting['plot_sc_verification'] = False
    plotting['plot_2sd'] = False
    plotting['plot_quadrant'] = False
    return plotting


def default_runner_config() -> Dict[str, Any]:
    default_data_path = repo_paths.PATH_EXAMPLE_DATA
    if not os.path.isdir(default_data_path):
        default_data_path = os.path.join(repo_paths.PATH_RAW_DATA, 'data', 'EX27_180924_modified')

    return {
        'paths': {
            'data_path': default_data_path,
            'output_path': os.path.join(repo_paths.PATH_AUTOCYTOTOX_OUTPUT, 'example_run'),
            'manual_excel': None,
        },
        'workflow': {
            'method': 'axis',
            'effector_cell': 'NK',
            'target_cell': 'K562',
            'identifier_strategy': 'well_row',
            'identifier_callable': None,
            'mwe': False,
            'mwe_n': 3,
            'mwe_seed': 42,
            'run_analysis': False,
            'export_csv': False,
            'min_analysis_cytotox': DEFAULT_ANALYSIS_MIN_CYTOTOX,
            'max_analysis_cv': DEFAULT_ANALYSIS_MAX_CV,
            'apply_analysis_cutoffs': True,
        },
        'channels': asdict(ChannelConfig()),
        'gating': asdict(GatingConfig()),
        'plotting': _default_plotting_config(),
        'runtime': {
            'quiet': False,
        },
    }


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_repo_path(path_value: Optional[str]) -> Optional[str]:
    if path_value in (None, ''):
        return None
    normalized_path = str(path_value)
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.join(repo_paths.PATH_RAW_DATA, normalized_path)


def _require_mapping(config_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = config_data.get(section)
    if not isinstance(value, dict):
        raise ValueError(f'Config section {section!r} must be a mapping.')
    return value


def _disable_all_plots(plotting: Dict[str, Any]) -> None:
    plotting['enable_all'] = False
    for key in PLOT_TOGGLE_KEYS:
        plotting[key] = False


def _enable_all_plots(plotting: Dict[str, Any]) -> None:
    plotting['enable_all'] = True
    for key in PLOT_TOGGLE_KEYS:
        plotting[key] = True


def validate_runner_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    paths = _require_mapping(config_data, 'paths')
    workflow = _require_mapping(config_data, 'workflow')
    channels = _require_mapping(config_data, 'channels')
    gating = _require_mapping(config_data, 'gating')
    plotting = _require_mapping(config_data, 'plotting')
    runtime = _require_mapping(config_data, 'runtime')

    paths['data_path'] = _resolve_repo_path(paths.get('data_path'))
    paths['output_path'] = _resolve_repo_path(paths.get('output_path'))
    paths['manual_excel'] = _resolve_repo_path(paths.get('manual_excel'))

    if not paths['data_path']:
        raise ValueError('paths.data_path is required.')
    if not paths['output_path']:
        raise ValueError('paths.output_path is required.')

    method = workflow.get('method')
    if method not in VALID_METHODS:
        raise ValueError(f'workflow.method must be one of {sorted(VALID_METHODS)}.')

    workflow['identifier_strategy'] = workflow.get('identifier_strategy') or 'well_row'
    workflow['identifier_callable'] = workflow.get('identifier_callable') or None
    if workflow['identifier_callable'] is None:
        valid_identifier_strategies = get_builtin_identifier_strategy_names()
        if workflow['identifier_strategy'] not in valid_identifier_strategies:
            raise ValueError(
                f'workflow.identifier_strategy must be one of {valid_identifier_strategies} '
                'unless workflow.identifier_callable is provided.'
            )

    gate2_method = gating.get('gate2_method')
    if gate2_method not in VALID_GATE2_METHODS:
        raise ValueError(f'gating.gate2_method must be one of {sorted(VALID_GATE2_METHODS)}.')

    gate_derivation = gating.get('gate_derivation')
    if gate_derivation not in VALID_GATE_DERIVATIONS:
        raise ValueError(f'gating.gate_derivation must be one of {sorted(VALID_GATE_DERIVATIONS)}.')

    plot_format = plotting.get('figure_format')
    if plot_format not in VALID_PLOT_FORMATS:
        raise ValueError(f'plotting.figure_format must be one of {sorted(VALID_PLOT_FORMATS)}.')

    percent = float(gating.get('percent', 0))
    if not 0 < percent <= 1:
        raise ValueError('gating.percent must be in the interval (0, 1].')

    percentile_q = float(gating.get('percentile_q', 0))
    if not 0 < percentile_q <= 1:
        raise ValueError('gating.percentile_q must be in the interval (0, 1].')

    subsample = float(gating.get('subsample', 0))
    if not 0 <= subsample <= 1:
        raise ValueError('gating.subsample must be in the interval [0, 1].')

    beads_cutoff = gating.get('beads_x_cutoff')
    if beads_cutoff == 0:
        gating['beads_x_cutoff'] = None

    if plotting.get('enable_all', True):
        _enable_all_plots(plotting)
    else:
        _disable_all_plots(plotting)

    runtime['quiet'] = bool(runtime.get('quiet', False))
    workflow['apply_analysis_cutoffs'] = bool(workflow.get('apply_analysis_cutoffs', True))
    return config_data


def load_runner_config(config_path: str) -> Dict[str, Any]:
    resolved_path = _resolve_repo_path(config_path)
    if not resolved_path or not os.path.exists(resolved_path):
        raise FileNotFoundError(f'Runner config not found: {config_path}')

    with open(resolved_path, 'r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError('Runner config must deserialize to a mapping.')

    merged = _deep_update(default_runner_config(), loaded)
    return validate_runner_config(merged)


def apply_runner_overrides(
    config_data: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    manual_excel: Optional[str] = None,
    quiet: bool = False,
    no_plots: bool = False,
) -> Dict[str, Any]:
    overridden = deepcopy(config_data)

    if overrides:
        _deep_update(overridden, overrides)

    if data_path:
        overridden['paths']['data_path'] = data_path
    if output_path:
        overridden['paths']['output_path'] = output_path
    if manual_excel:
        overridden['paths']['manual_excel'] = manual_excel
    if quiet:
        overridden['runtime']['quiet'] = True
    if no_plots:
        _disable_all_plots(overridden['plotting'])

    return validate_runner_config(overridden)


def build_args_namespace(config_data: Dict[str, Any], config_path: Optional[str] = None) -> argparse.Namespace:
    paths = config_data['paths']
    workflow = config_data['workflow']
    gating = config_data['gating']
    plotting = config_data['plotting']
    runtime = config_data['runtime']

    return argparse.Namespace(
        config_path=config_path,
        data_path=paths['data_path'],
        output_path=paths['output_path'],
        manual_excel=paths['manual_excel'],
        method=workflow['method'],
        percent=gating['percent'],
        bw_adjust=gating['bw_adjust'],
        min_distance=gating['min_distance'],
        identifier_strategy=workflow['identifier_strategy'],
        identifier_callable=workflow['identifier_callable'],
        gate2_method=gating['gate2_method'],
        beads_x_cutoff=gating['beads_x_cutoff'],
        gate_derivation=gating['gate_derivation'],
        percentile_q=gating['percentile_q'],
        k_sigma=gating['k_sigma'],
        effector_cell=workflow['effector_cell'],
        target_cell=workflow['target_cell'],
        mwe=workflow['mwe'],
        mwe_n=workflow['mwe_n'],
        mwe_seed=workflow['mwe_seed'],
        subsample=gating['subsample'],
        min_analysis_cytotox=workflow['min_analysis_cytotox'],
        max_analysis_cv=workflow['max_analysis_cv'],
        apply_analysis_cutoffs=workflow['apply_analysis_cutoffs'],
        disable_analysis_cutoffs=not workflow['apply_analysis_cutoffs'],
        no_plots=not plotting['enable_all'],
        plot_format=plotting['figure_format'],
        run_analysis=workflow['run_analysis'],
        export_csv=workflow['export_csv'],
        verbose=not runtime['quiet'],
        quiet=runtime['quiet'],
    )


def build_runtime_configs(config_data: Dict[str, Any]):
    channels = ChannelConfig(**config_data['channels'])
    gating = GatingConfig(**config_data['gating'])
    plotting = PlotConfig(**config_data['plotting'])
    return channels, gating, plotting
