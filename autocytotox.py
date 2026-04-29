#!/usr/bin/env python3
"""autocytotox - YAML-driven autogating cytotoxicity analysis (recommended entrypoint).

Recommended use for scientists
------------------------------

    1. Copy examples/simple_example_config.yaml to my_config.yaml
    2. Edit paths.data_path, paths.output_path and channels.* to match your data
    3. Run:
           python autocytotox.py --config my_config.yaml

Open examples/simple_example.ipynb for an interactive walkthrough.

Advanced users can override any YAML field from the CLI; run
``python autocytotox.py --help`` for the full list.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.cli import main


if __name__ == '__main__':
    main()
