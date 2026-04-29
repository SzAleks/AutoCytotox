#!/usr/bin/env python3
"""
autocytotox_gridsearch.py - Run a grid search over the autocytotox autogating workflow.

For each combination of grid parameters in the YAML config, this script runs
the full autocytotox pipeline and writes results to a dedicated subfolder under
``paths.output_path``.

Usage
-----
    python autocytotox_gridsearch.py                          # uses gridsearch_config.yaml
    python autocytotox_gridsearch.py --config my_grid.yaml    # custom config
    python autocytotox_gridsearch.py --yes                    # skip the [y/N] prompt
    python autocytotox_gridsearch.py --dry-run                # only expand + preview

WARNING
-------
A full grid (many methods x percents x bandwidths x ...) can take HOURS.
The script always shows the expansion count and asks for confirmation
before running. The default ``gridsearch_config.yaml`` in this repo is a
small smoke-test grid (1 configuration).
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src import paths as repo_paths
from src.gridsearch import (
    configuration_label,
    expand_gridsearch_configurations,
    load_gridsearch_config,
    run_gridsearch,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='autocytotox grid search - run the full pipeline for every grid combination',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=repo_paths.DEFAULT_GRIDSEARCH_CONFIG,
        help='Path to the YAML gridsearch configuration file',
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip the interactive [y/N] confirmation prompt',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only expand and preview the grid; do not run the pipeline',
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Re-run configurations even if their output Excel already exists',
    )
    parser.add_argument(
        '--show-first',
        type=int,
        default=5,
        help='Number of expanded configurations to print as a preview',
    )
    return parser.parse_args(argv)


def _preview(config_data, configurations, show_first):
    print(f"Gridsearch config: {config_data.get('__source_path__', '<resolved>')}")
    print(f"Data path:   {config_data['paths']['data_path']}")
    print(f"Output path: {config_data['paths']['output_path']}")
    print(f"Expanded configurations: {len(configurations)}")

    preview_count = max(0, min(show_first, len(configurations)))
    if preview_count:
        total = len(configurations)
        print('Preview:')
        for index, cfg in enumerate(configurations[:preview_count], 1):
            print(f"  {configuration_label(index, total, cfg)}: {cfg}")


def main(argv=None):
    args = parse_args(argv)

    config_data = load_gridsearch_config(args.config)
    configurations = expand_gridsearch_configurations(config_data)
    _preview(config_data, configurations, args.show_first)

    if args.dry_run:
        return {'config': config_data, 'configurations': configurations}

    resume_flag = False if args.no_resume else None
    summary = run_gridsearch(
        config_path=args.config,
        yes=args.yes,
        resume=resume_flag,
    )
    return summary


if __name__ == '__main__':
    main()
