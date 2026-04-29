"""Gridsearch configuration tests for the AutoCytotox workflow.

These tests intentionally validate grid expansion and runner-config generation
without running every grid entry end to end.
"""

from __future__ import annotations

import itertools
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDSEARCH_CONFIG = REPO_ROOT / 'gridsearch_config.yaml'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class GridsearchConfigurationTests(unittest.TestCase):
    """Checks for the supported gridsearch method families."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_root = Path(tempfile.mkdtemp(prefix='autocytotox_gridsearch_'))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp_root, ignore_errors=True)

    def test_gridsearch_smoke_config_matches_validated_best_workflow(self) -> None:
        from src.gridsearch import (
            _build_runner_config_for_entry,
            expand_gridsearch_configurations,
            load_gridsearch_config,
        )

        config = load_gridsearch_config(str(GRIDSEARCH_CONFIG))
        entries = expand_gridsearch_configurations(config)

        self.assertEqual(
            entries,
            [{
                'method': 'euclidean',
                'percent': 0.95,
                'bw_adjust': 0.05,
                'min_distance': 30,
                'gate2_method': 'ratio',
                'gate_derivation': 'percentile',
                'percentile_q': 0.90,
                'k_sigma': 2.5,
            }],
        )

        runner = _build_runner_config_for_entry(config, entries[0], str(self._tmp_root / 'best_workflow'))
        self.assertEqual(runner['workflow']['method'], 'euclidean')
        self.assertEqual(runner['gating']['percent'], 0.95)
        self.assertEqual(runner['gating']['bw_adjust'], 0.05)
        self.assertEqual(runner['gating']['min_distance'], 30)
        self.assertEqual(runner['gating']['gate2_method'], 'ratio')
        self.assertEqual(runner['gating']['gate_derivation'], 'percentile')
        self.assertEqual(runner['gating']['percentile_q'], 0.90)
        self.assertEqual(runner['gating']['beads_x_cutoff'], 2400000)
        self.assertFalse(runner['plotting']['enable_all'])

    def test_all_supported_method_families_build_valid_runner_configs(self) -> None:
        from src.config_loader import VALID_GATE2_METHODS, VALID_GATE_DERIVATIONS, VALID_METHODS
        from src.gridsearch import (
            _build_runner_config_for_entry,
            configuration_label,
            expand_gridsearch_configurations,
            load_gridsearch_config,
        )

        config = load_gridsearch_config(str(GRIDSEARCH_CONFIG))
        config['grid'] = {
            'method': sorted(VALID_METHODS),
            'percent': [0.95],
            'bw_adjust': [0.05],
            'min_distance': [30],
            'gate2_method': sorted(VALID_GATE2_METHODS),
            'gate_derivation': sorted(VALID_GATE_DERIVATIONS),
            'percentile_q': [0.90],
            'k_sigma': [2.5],
        }

        entries = expand_gridsearch_configurations(config)
        expected_method_grid = set(itertools.product(VALID_METHODS, VALID_GATE2_METHODS, VALID_GATE_DERIVATIONS))
        observed_method_grid = {
            (entry['method'], entry['gate2_method'], entry['gate_derivation'])
            for entry in entries
        }

        self.assertEqual(len(entries), 27)
        self.assertEqual(observed_method_grid, expected_method_grid)

        labels = set()
        for index, entry in enumerate(entries, start=1):
            label = configuration_label(index, len(entries), entry)
            self.assertNotIn(label, labels)
            labels.add(label)

            runner = _build_runner_config_for_entry(
                config,
                entry,
                str(self._tmp_root / 'all_methods' / label),
            )

            self.assertEqual(runner['workflow']['method'], entry['method'])
            self.assertEqual(runner['gating']['gate2_method'], entry['gate2_method'])
            self.assertEqual(runner['gating']['gate_derivation'], entry['gate_derivation'])
            self.assertEqual(runner['gating']['percent'], 0.95)
            self.assertEqual(runner['gating']['bw_adjust'], 0.05)
            self.assertEqual(runner['gating']['min_distance'], 30)
            self.assertEqual(runner['gating']['beads_x_cutoff'], 2400000)

            if entry['gate_derivation'] == 'percentile':
                self.assertEqual(runner['gating']['percentile_q'], 0.90)
            else:
                self.assertEqual(runner['gating']['percentile_q'], 0.99)

            if entry['gate_derivation'] == 'mean_ksd':
                self.assertEqual(runner['gating']['k_sigma'], 2.5)


if __name__ == '__main__':
    unittest.main()
