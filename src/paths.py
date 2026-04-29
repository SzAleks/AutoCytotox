"""
Repository Path Constants for the autocytotox Autogating Pipeline.

This module centralizes the canonical filesystem locations used by the
pipeline (example data, default YAML configurations, output root) so that
other modules can reference them without hard-coding relative paths.

Author: Aleksander Szarzynski TUW 2026
"""

import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

PATH_RAW_DATA = os.path.join(parent_dir)
PATH_EXAMPLE_DATA = os.path.join(parent_dir, 'example_data')
PATH_EXAMPLES = os.path.join(parent_dir, 'examples')
PATH_AUTOCYTOTOX_OUTPUT = os.path.join(parent_dir, 'output', 'autocytotox')
DEFAULT_RUNNER_CONFIG = os.path.join(parent_dir, 'config.yaml')
DEFAULT_GRIDSEARCH_CONFIG = os.path.join(parent_dir, 'gridsearch_config.yaml')
DEFAULT_EXAMPLE_CONFIG = os.path.join(PATH_EXAMPLES, 'simple_example_config.yaml')

# Manual analyst Excel comparison is opt-in. Set it explicitly via either:
#   - paths.manual_excel in your YAML config, or
#   - the --manual-excel CLI flag.
# No default is provided here because manual analyst spreadsheets typically
# contain non-anonymized data (e.g. analyst initials in the filename) and
# should not be hard-coded into a published repository.

