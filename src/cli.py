"""
Command-Line Interface for the autocytotox Autogating Pipeline.

This module defines the argparse parser used by ``autocytotox.py``. It loads
the YAML runner configuration, applies optional CLI overrides for individual
fields, and dispatches to ``runner.run_from_config``.

Author: Aleksander Szarzynski TUW 2026
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable

from . import paths as repo_paths
from .config_loader import apply_runner_overrides, load_runner_config
from .identifiers import get_builtin_identifier_strategy_names
from .workflow.runner import run_from_config


def _default_example_output_path() -> str:
    return os.path.join(repo_paths.PATH_AUTOCYTOTOX_OUTPUT, 'example_run')


def _default_enabled_plotting(figure_format: str | None = None) -> Dict[str, Any]:
    plotting: Dict[str, Any] = {
        'enable_all': True,
        'plot_beads': True,
        'plot_gate_lines': True,
        'plot_proximity': True,
        'plot_kde_minima': True,
        'plot_sc_verification': True,
        'plot_2sd': True,
        'plot_quadrant': True,
    }
    if figure_format:
        plotting['figure_format'] = figure_format
    return plotting


@dataclass(frozen=True)
class OverrideOption:
    flags: tuple[str, ...]
    dest: str
    section: str
    key: str
    argparse_kwargs: Dict[str, Any] = field(default_factory=dict)
    is_active: Callable[[Any], bool] | None = None
    transform: Callable[[Any], Any] | None = None


OVERRIDE_OPTIONS: tuple[OverrideOption, ...] = (
    OverrideOption(('--data-path',), 'data_path', 'paths', 'data_path', {'type': str, 'default': None, 'help': 'Optional override for the configured data path'}),
    OverrideOption(('--output-path',), 'output_path', 'paths', 'output_path', {'type': str, 'default': None, 'help': 'Optional override for the configured output path'}),
    OverrideOption(('--manual-excel',), 'manual_excel', 'paths', 'manual_excel', {'type': str, 'default': None, 'help': 'Optional override for the configured manual Excel path'}),
    OverrideOption(('--method',), 'method', 'workflow', 'method', {'type': str, 'choices': ['axis', 'euclidean', 'density'], 'default': None, 'help': 'Override the proximity gating method used for cytotoxicity evaluation'}),
    OverrideOption(('--effector-cell',), 'effector_cell', 'workflow', 'effector_cell', {'type': str, 'default': None, 'help': 'Override the effector-cell filename identifier'}),
    OverrideOption(('--target-cell',), 'target_cell', 'workflow', 'target_cell', {'type': str, 'default': None, 'help': 'Override the target-cell filename identifier'}),
    OverrideOption(('--identifier-strategy',), 'identifier_strategy', 'workflow', 'identifier_strategy', {'type': str, 'choices': get_builtin_identifier_strategy_names(), 'default': None, 'help': 'Built-in identifier grouping strategy used to map filenames to experimental groups'}),
    OverrideOption(('--identifier-callable',), 'identifier_callable', 'workflow', 'identifier_callable', {'type': str, 'default': None, 'help': 'Custom identifier handler in the form module:function or path/to/file.py:function'}),
    OverrideOption(('--mwe',), 'mwe', 'workflow', 'mwe', {'action': 'store_true', 'help': 'Process only a subset of day folders'}, is_active=bool),
    OverrideOption(('--mwe-n',), 'mwe_n', 'workflow', 'mwe_n', {'type': int, 'default': None, 'help': 'Number of folders to process in MWE mode'}),
    OverrideOption(('--mwe-seed',), 'mwe_seed', 'workflow', 'mwe_seed', {'type': int, 'default': None, 'help': 'Random seed for MWE folder selection'}),
    OverrideOption(('--run-analysis',), 'run_analysis', 'workflow', 'run_analysis', {'action': 'store_true', 'help': 'Run extended postanalysis after the main pipeline'}, is_active=bool),
    OverrideOption(('--export-csv',), 'export_csv', 'workflow', 'export_csv', {'action': 'store_true', 'help': 'Export CSV tables during postanalysis'}, is_active=bool),
    OverrideOption(('--min-analysis-cytotox',), 'min_analysis_cytotox', 'workflow', 'min_analysis_cytotox', {'type': float, 'default': None, 'help': 'Override the minimum cytotoxicity threshold used for downstream analytics'}),
    OverrideOption(('--max-analysis-cv',), 'max_analysis_cv', 'workflow', 'max_analysis_cv', {'type': float, 'default': None, 'help': 'Override the maximum CV threshold used for downstream analytics'}),
    OverrideOption(('--disable-analysis-cutoffs',), 'disable_analysis_cutoffs', 'workflow', 'apply_analysis_cutoffs', {'action': 'store_true', 'help': 'Disable the downstream analytics CV and cytotoxicity cutoffs'}, is_active=bool, transform=lambda _value: False),
    OverrideOption(('--percent',), 'percent', 'gating', 'percent', {'type': float, 'default': None, 'help': 'Override the proximity gating percent cutoff'}),
    OverrideOption(('--bw-adjust',), 'bw_adjust', 'gating', 'bw_adjust', {'type': float, 'default': None, 'help': 'Override the KDE bandwidth adjustment'}),
    OverrideOption(('--min-distance',), 'min_distance', 'gating', 'min_distance', {'type': int, 'default': None, 'help': 'Override the minimum distance between KDE peaks'}),
    OverrideOption(('--gate2-method',), 'gate2_method', 'gating', 'gate2_method', {'type': str, 'choices': ['knn', 'ratio', 'linear_band'], 'default': None, 'help': 'Override the gate-2 singlet or morphology gate'}),
    OverrideOption(('--beads-x-cutoff',), 'beads_x_cutoff', 'gating', 'beads_x_cutoff', {'type': float, 'default': None, 'help': 'Override the beads FSC-A cutoff. The default is a hard-coded numeric value (recommended). Pass 0 to enable OPTIONAL auto-derivation from a Size Beads .fcs file -- only valid if size-calibration beads were actually measured on the plate; noisy bead signals can shift the auto cutoff, so use at your own risk.'}, transform=lambda value: None if value == 0 else value),
    OverrideOption(('--gate-derivation',), 'gate_derivation', 'gating', 'gate_derivation', {'type': str, 'choices': ['max', 'percentile', 'mean_ksd'], 'default': None, 'help': 'Override how quadrant gates are derived from the reference population'}),
    OverrideOption(('--percentile-q',), 'percentile_q', 'gating', 'percentile_q', {'type': float, 'default': None, 'help': 'Override the quantile used for percentile-based gate derivation'}),
    OverrideOption(('--k-sigma',), 'k_sigma', 'gating', 'k_sigma', {'type': float, 'default': None, 'help': 'Override the number of SDs used for mean_ksd gate derivation'}),
    OverrideOption(('--subsample',), 'subsample', 'gating', 'subsample', {'type': float, 'default': None, 'help': 'Override the per-file event subsample fraction'}),
)


def _compact_overrides(overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {section: values for section, values in overrides.items() if values}


def _iter_override_sections() -> Iterable[str]:
    return ('paths', 'workflow', 'gating', 'plotting', 'runtime')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'autocytotox - YAML-driven autogating cytotoxicity analysis. '
            'For most users, the recommended workflow is to copy '
            'examples/simple_example_config.yaml, edit the three highlighted '
            'fields (paths.data_path, paths.output_path, channels.*) to point '
            'at your data, then run: python autocytotox.py --config my_config.yaml. '
            'The CLI flags below let advanced users override individual YAML fields without '
            'editing the file.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=repo_paths.DEFAULT_RUNNER_CONFIG,
        help='Path to the YAML runner configuration file',
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Force the committed example dataset and example output path',
    )
    parser.add_argument(
        '--no-manual',
        action='store_true',
        help='Disable manual Excel loading even if configured in YAML',
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Enable the default plot set without editing the YAML file',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose progress output',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Force verbose progress output even if the YAML file is quiet',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation without changing the YAML file',
    )
    parser.add_argument(
        '--plot-format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default=None,
        help='Override the plot file format',
    )
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Print the resolved runtime configuration before execution',
    )

    for option in OVERRIDE_OPTIONS:
        parser.add_argument(*option.flags, dest=option.dest, **option.argparse_kwargs)

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _build_override_map(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {section: {} for section in _iter_override_sections()}

    if args.example:
        overrides['paths']['data_path'] = repo_paths.PATH_EXAMPLE_DATA
        if not args.output_path:
            overrides['paths']['output_path'] = _default_example_output_path()

    for option in OVERRIDE_OPTIONS:
        value = getattr(args, option.dest)
        if option.is_active is not None:
            if not option.is_active(value):
                continue
        elif value is None:
            continue

        if option.transform is not None:
            value = option.transform(value)

        overrides[option.section][option.key] = value

    if args.no_manual:
        overrides['paths']['manual_excel'] = None

    if args.plots:
        overrides['plotting'].update(_default_enabled_plotting(args.plot_format))
    elif args.plot_format:
        overrides['plotting']['figure_format'] = args.plot_format

    if args.verbose:
        overrides['runtime']['quiet'] = False
    if args.quiet:
        overrides['runtime']['quiet'] = True

    return _compact_overrides(overrides)


def main(argv=None):
    args = parse_args(argv)
    config_data = load_runner_config(args.config)
    override_map = _build_override_map(args)
    config_data = apply_runner_overrides(
        config_data,
        overrides=override_map,
        data_path=args.data_path,
        output_path=args.output_path,
        quiet=args.quiet,
        no_plots=args.no_plots,
    )

    if args.show_config:
        print(json.dumps(config_data, indent=2, sort_keys=True))

    return run_from_config(config_data, config_path=args.config)
