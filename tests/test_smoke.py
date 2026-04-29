"""Smoke tests for the AutoCytotox publication workflow.

Run with:
    python -m unittest discover -s tests -p "test_*.py"
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = REPO_ROOT / 'example_data'
EXAMPLE_CONFIG = REPO_ROOT / 'examples' / 'simple_example_config.yaml'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class AutoCytotoxSmokeTests(unittest.TestCase):
    """End-to-end smoke checks for the bundled example data."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.config_loader import apply_runner_overrides, load_runner_config
        from src.workflow.runner import run_from_config

        cls._tmp_root = Path(tempfile.mkdtemp(prefix='autocytotox_smoke_'))
        cls.output_dir = cls._tmp_root / 'run'

        config = load_runner_config(str(EXAMPLE_CONFIG))
        runtime = apply_runner_overrides(
            config,
            overrides={'plotting': {'enable_all': False}},
            output_path=str(cls.output_dir),
            quiet=True,
        )
        cls.results = run_from_config(runtime, config_path=str(EXAMPLE_CONFIG))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp_root, ignore_errors=True)

    def test_example_data_present(self) -> None:
        self.assertTrue(EXAMPLE_DATA.is_dir(), f'Missing example_data at {EXAMPLE_DATA}')
        day_folders = sorted(p.name for p in EXAMPLE_DATA.iterdir() if p.is_dir())
        self.assertGreaterEqual(len(day_folders), 5, f'Expected at least 5 Day folders, got {day_folders}')

    def test_example_config_loads(self) -> None:
        from src.config_loader import load_runner_config

        config = load_runner_config(str(EXAMPLE_CONFIG))
        self.assertEqual(config['workflow']['method'], 'euclidean')
        self.assertEqual(config['gating']['percent'], 0.95)
        self.assertEqual(config['gating']['bw_adjust'], 0.05)
        self.assertEqual(config['gating']['min_distance'], 30)
        self.assertEqual(config['gating']['gate2_method'], 'ratio')
        self.assertEqual(config['gating']['gate_derivation'], 'percentile')
        self.assertEqual(config['gating']['percentile_q'], 0.90)
        self.assertEqual(config['gating']['beads_x_cutoff'], 2400000)

    def test_autocytotox_runs_end_to_end(self) -> None:
        raw = self.results['combined_results']
        self.assertIsInstance(raw, pd.DataFrame)

        cytotox = raw.loc[raw['Cytotoxicity'].notna(), ['Folder', 'File', 'Cytotoxicity']]
        self.assertGreaterEqual(len(cytotox), 30, f'Expected >=30 cytotox values, got {len(cytotox)}')
        self.assertGreaterEqual(cytotox['Folder'].nunique(), 5)

    def test_cytotoxicity_in_plausible_range(self) -> None:
        raw = self.results['combined_results']
        cytotox = raw['Cytotoxicity'].dropna()
        self.assertTrue(
            cytotox.between(-100, 100).all(),
            f'Out-of-range cytotox values: min={cytotox.min()}, max={cytotox.max()}',
        )
        self.assertTrue(math.isfinite(cytotox.mean()))

    def test_aggregate_groups_have_cv(self) -> None:
        agg = self.results['auto_agg']
        self.assertIsInstance(agg, pd.DataFrame)
        self.assertTrue({'cytotoxicity_mean', 'cv_percent', 'n_values'}.issubset(agg.columns))
        self.assertTrue((agg['n_values'] > 0).all())

    def test_excel_output_written(self) -> None:
        excels = sorted(self.output_dir.glob('autocytotox_results_*.xlsx'))
        self.assertEqual(len(excels), 1, f'Expected exactly one Excel output, got {excels}')
        self.assertGreater(excels[0].stat().st_size, 0)

    def test_determinism_two_runs(self) -> None:
        from src.config_loader import apply_runner_overrides, load_runner_config
        from src.workflow.runner import run_from_config

        config = load_runner_config(str(EXAMPLE_CONFIG))
        tmp = Path(tempfile.mkdtemp(prefix='autocytotox_determinism_'))

        def _run(out_dir: Path) -> pd.DataFrame:
            runtime = apply_runner_overrides(
                config,
                overrides={'plotting': {'enable_all': False}},
                output_path=str(out_dir),
                quiet=True,
            )
            results = run_from_config(runtime, config_path=str(EXAMPLE_CONFIG))
            return (
                results['combined_results']
                .loc[lambda df: df['Cytotoxicity'].notna(), ['Folder', 'File', 'Cytotoxicity']]
                .sort_values(['Folder', 'File'])
                .reset_index(drop=True)
            )

        try:
            a = _run(tmp / 'a')
            b = _run(tmp / 'b')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(list(a.columns), list(b.columns))
        self.assertEqual(len(a), len(b))
        pd.testing.assert_series_equal(
            a['Cytotoxicity'],
            b['Cytotoxicity'],
            check_exact=False,
            atol=1e-9,
            rtol=0,
        )


if __name__ == '__main__':
    unittest.main()
