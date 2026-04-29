"""
Hyperparameter Grid Search Engine for the autocytotox Autogating Pipeline.

This module expands a YAML grid configuration into individual parameter
combinations and runs the full pipeline once per combination with resume
support. Downstream ranking can be performed from the generated result
workbooks.

Author: Aleksander Szarzynski TUW 2026
"""

import itertools
import os
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
except ImportError as exc:
    raise RuntimeError('PyYAML is required for the autocytotox gridsearch configuration workflow.') from exc

from .. import paths as repo_paths
from ..config_loader import (
    VALID_GATE2_METHODS,
    VALID_GATE_DERIVATIONS,
    VALID_METHODS,
    _deep_update,
    default_runner_config,
    validate_runner_config,
)


def _resolve_repo_path(path_value: str | None) -> str | None:
    if path_value is None or path_value == '':
        return None
    normalized_path: str = str(path_value)
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.join(repo_paths.PATH_RAW_DATA, normalized_path)


def load_gridsearch_config(config_path: str) -> Dict[str, Any]:
    resolved_path = _resolve_repo_path(config_path)
    if not resolved_path or not os.path.exists(resolved_path):
        raise FileNotFoundError(f'Gridsearch config not found: {config_path}')

    with open(resolved_path, 'r', encoding='utf-8') as handle:
        config_data = yaml.safe_load(handle) or {}

    if not isinstance(config_data, dict):
        raise ValueError('Gridsearch config must deserialize to a mapping.')

    for section in ['paths', 'workflow', 'grid', 'runtime']:
        if not isinstance(config_data.get(section), dict):
            raise ValueError(f'Gridsearch config section {section!r} must be a mapping.')

    config_data['paths']['data_path'] = _resolve_repo_path(config_data['paths'].get('data_path'))
    config_data['paths']['output_path'] = _resolve_repo_path(config_data['paths'].get('output_path'))
    config_data['paths']['manual_excel'] = _resolve_repo_path(config_data['paths'].get('manual_excel'))

    if not config_data['paths']['data_path']:
        raise ValueError('paths.data_path is required.')
    if not config_data['paths']['output_path']:
        raise ValueError('paths.output_path is required.')

    grid = config_data['grid']
    for key in ['method', 'percent', 'bw_adjust', 'min_distance', 'gate2_method', 'gate_derivation', 'percentile_q', 'k_sigma']:
        if key not in grid or not isinstance(grid[key], list) or not grid[key]:
            raise ValueError(f'grid.{key} must be a non-empty list.')

    if any(method not in VALID_METHODS for method in grid['method']):
        raise ValueError(f'grid.method must contain only {sorted(VALID_METHODS)}.')
    if any(method not in VALID_GATE2_METHODS for method in grid['gate2_method']):
        raise ValueError(f'grid.gate2_method must contain only {sorted(VALID_GATE2_METHODS)}.')
    if any(method not in VALID_GATE_DERIVATIONS for method in grid['gate_derivation']):
        raise ValueError(f'grid.gate_derivation must contain only {sorted(VALID_GATE_DERIVATIONS)}.')

    return config_data


def expand_gridsearch_configurations(config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = config_data['grid']
    configurations = []

    for method, percent, bw_adjust, min_distance, gate2_method, gate_derivation, percentile_q, k_sigma in itertools.product(
        grid['method'],
        grid['percent'],
        grid['bw_adjust'],
        grid['min_distance'],
        grid['gate2_method'],
        grid['gate_derivation'],
        grid['percentile_q'],
        grid['k_sigma'],
    ):
        configuration = {
            'method': method,
            'percent': percent,
            'bw_adjust': bw_adjust,
            'min_distance': min_distance,
            'gate2_method': gate2_method,
            'gate_derivation': gate_derivation,
            'percentile_q': percentile_q if gate_derivation == 'percentile' else 0.99,
            'k_sigma': k_sigma if gate_derivation == 'mean_ksd' else 2.5,
        }
        configurations.append(configuration)

    return configurations


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

GRID_KEYS = (
    'method', 'percent', 'bw_adjust', 'min_distance',
    'gate2_method', 'gate_derivation', 'percentile_q', 'k_sigma',
)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        text = f'{value:g}'
        return text.replace('.', 'p')
    return str(value)


def configuration_label(index: int, total: int, configuration: Dict[str, Any]) -> str:
    """Return a deterministic, filesystem-safe folder name for one grid entry."""
    width = max(3, len(str(total)))
    parts = [
        f'{index:0{width}d}',
        f"m={_format_value(configuration['method'])}",
        f"p={_format_value(configuration['percent'])}",
        f"bw={_format_value(configuration['bw_adjust'])}",
        f"md={_format_value(configuration['min_distance'])}",
        f"g2={_format_value(configuration['gate2_method'])}",
        f"gd={_format_value(configuration['gate_derivation'])}",
    ]
    if configuration['gate_derivation'] == 'percentile':
        parts.append(f"q={_format_value(configuration['percentile_q'])}")
    elif configuration['gate_derivation'] == 'mean_ksd':
        parts.append(f"ks={_format_value(configuration['k_sigma'])}")
    return '__'.join(parts)


def _build_runner_config_for_entry(
    grid_config: Dict[str, Any],
    entry: Dict[str, Any],
    output_subdir: str,
) -> Dict[str, Any]:
    base = default_runner_config()

    paths = {
        'data_path': grid_config['paths']['data_path'],
        'output_path': output_subdir,
        'manual_excel': grid_config['paths'].get('manual_excel'),
    }

    workflow = dict(grid_config.get('workflow') or {})
    workflow['method'] = entry['method']

    gating = dict(grid_config.get('gating_defaults') or {})
    for key in GRID_KEYS:
        if key == 'method':
            continue  # 'method' belongs in workflow, set above
        gating[key] = entry[key]

    runtime = dict(grid_config.get('runtime') or {})
    # Strip gridsearch-only keys that the runner does not consume.
    runtime.pop('n_workers', None)
    runtime.pop('resume', None)

    overrides = {
        'paths': paths,
        'workflow': workflow,
        'channels': dict(grid_config.get('channels') or {}),
        'gating': gating,
        'plotting': dict(grid_config.get('plotting') or {}),
        'runtime': runtime,
    }
    _deep_update(base, overrides)
    return validate_runner_config(base)


def _confirm_run(total: int, yes: bool) -> bool:
    if yes:
        return True
    print('')
    print('=' * 60)
    print('  GRID SEARCH WARNING')
    print('=' * 60)
    print(f'  About to run {total} configuration(s) end-to-end.')
    print('  Each configuration runs the full autocytotox pipeline (folder discovery,')
    print('  beads/proximity/quadrant gating, plotting, Excel export).')
    print('  Runtime scales linearly with the number of configurations and')
    print('  with the size of your data folder. Large grids can take HOURS.')
    print('=' * 60)
    try:
        answer = input(f'  Continue with all {total} configuration(s)? [y/N]: ')
    except EOFError:
        answer = ''
    return answer.strip().lower() in ('y', 'yes')


def run_gridsearch(
    config_path: str = repo_paths.DEFAULT_GRIDSEARCH_CONFIG,
    yes: bool = False,
    resume: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Execute every expanded configuration from a gridsearch YAML.

    Parameters
    ----------
    config_path
        Path to the gridsearch YAML (defaults to ``gridsearch_config.yaml``).
    yes
        Skip the interactive ``[y/N]`` confirmation. Useful from notebooks
        and CI.
    resume
        If True, skip configurations whose output folder already contains a
        results Excel. If None (default), the value is taken from
        ``runtime.resume`` in the YAML and falls back to True.

    Returns
    -------
    dict | None
        ``{'configurations': [...], 'results': [...]}`` on success, ``None``
        if the user aborts at the confirmation prompt.
    """
    grid_config = load_gridsearch_config(config_path)
    configurations = expand_gridsearch_configurations(grid_config)
    total = len(configurations)

    if total == 0:
        print('No configurations expanded from grid; nothing to run.')
        return {'configurations': [], 'results': []}

    if resume is None:
        runtime_section = grid_config.get('runtime') or {}
        resume = bool(runtime_section.get('resume', True))

    if not _confirm_run(total, yes):
        print('Aborted by user.')
        return None

    base_output = grid_config['paths']['output_path']
    os.makedirs(base_output, exist_ok=True)

    results: List[Dict[str, Any]] = []
    grid_started = time.time()

    for index, entry in enumerate(configurations, 1):
        label = configuration_label(index, total, entry)
        out_dir = os.path.join(base_output, label)

        elapsed_grid = time.time() - grid_started
        print('')
        print('-' * 60)
        print(f'[{index}/{total}] {label}   (grid elapsed: {elapsed_grid:.1f}s)')
        print(f'  config: {entry}')
        print(f'  output: {out_dir}')

        existing_excel = _find_existing_excel(out_dir)
        if resume and existing_excel:
            print(f'  SKIP (resume): {existing_excel}')
            results.append({
                'index': index,
                'configuration': entry,
                'output_path': out_dir,
                'status': 'skipped',
                'excel_path': existing_excel,
            })
            continue

        try:
            runner_config = _build_runner_config_for_entry(grid_config, entry, out_dir)
            run_result = _run_one(runner_config)
            results.append({
                'index': index,
                'configuration': entry,
                'output_path': out_dir,
                'status': 'ok',
                'excel_path': run_result.get('excel_path'),
            })
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            print(f'  ERROR: {exc}')
            traceback.print_exc()
            results.append({
                'index': index,
                'configuration': entry,
                'output_path': out_dir,
                'status': 'failed',
                'error': str(exc),
            })

    total_elapsed = time.time() - grid_started
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    n_skip = sum(1 for r in results if r['status'] == 'skipped')
    n_fail = sum(1 for r in results if r['status'] == 'failed')
    print('')
    print('=' * 60)
    print(f'GRID SEARCH COMPLETE in {total_elapsed:.1f}s')
    print(f'  ok:      {n_ok}/{total}')
    print(f'  skipped: {n_skip}/{total}')
    print(f'  failed:  {n_fail}/{total}')
    print(f'  output:  {base_output}')
    print('=' * 60)

    return {
        'configurations': configurations,
        'results': results,
        'output_path': base_output,
    }


def _find_existing_excel(out_dir: str) -> Optional[str]:
    if not os.path.isdir(out_dir):
        return None
    for name in sorted(os.listdir(out_dir)):
        if name.lower().endswith('.xlsx') and not name.startswith('~$'):
            return os.path.join(out_dir, name)
    return None


def _run_one(runner_config: Dict[str, Any]) -> Dict[str, Any]:
    # Imported lazily to avoid pulling heavy modules when only validating
    # the gridsearch YAML.
    from ..workflow.runner import run_from_config
    return run_from_config(deepcopy(runner_config))
