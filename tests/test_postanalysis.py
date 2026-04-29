"""Postanalysis smoke tests using synthetic Excel inputs."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CONDITIONS = ('B', 'F', 'R')
OPT_REFS = ('Opt', 'Ref')
ANALYSTS = ('User 1', 'User 2', 'User 3')


def _synthetic_value(day: int, condition: str, opt_ref: str, bio_rep: int, analyst_index: int = 0) -> float:
    condition_offset = {'B': 0.0, 'F': 4.0, 'R': 8.0}[condition]
    opt_ref_offset = 0.0 if opt_ref == 'Opt' else 2.5
    return 18.0 + day * 1.4 + condition_offset + opt_ref_offset + bio_rep * 0.6 + analyst_index * 0.25


def _write_results_excel(path: Path) -> None:
    records = []
    for day in range(9):
        for condition in CONDITIONS:
            folder = f'Day {day}_{condition}'
            for opt_ref in OPT_REFS:
                for bio_rep in (1, 2, 3):
                    value = _synthetic_value(day, condition, opt_ref, bio_rep)
                    records.append({
                        'Folder': folder,
                        'File': f'05-CC_{condition}_{opt_ref}_{bio_rep}_1-A{bio_rep}.fcs',
                        'Cytotoxicity': value,
                    })

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame(records).to_excel(writer, sheet_name='Raw_Results', index=False)


def _write_manual_excel(path: Path) -> None:
    condition_cols = {
        'B': {'tech1': 4, 'tech2': 5, 'tech3': 6, 'mean': 7, 'std': 8},
        'F': {'tech1': 12, 'tech2': 13, 'tech3': 14, 'mean': 15, 'std': 16},
        'R': {'tech1': 20, 'tech2': 21, 'tech3': 22, 'mean': 23, 'std': 24},
    }
    bio_rep_starts = {
        ('Opt', 1): 2,
        ('Opt', 2): 12,
        ('Opt', 3): 22,
        ('Ref', 1): 33,
        ('Ref', 2): 43,
        ('Ref', 3): 53,
    }

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for analyst_index, analyst in enumerate(ANALYSTS):
            sheet = pd.DataFrame(np.nan, index=range(62), columns=range(25))
            for (opt_ref, bio_rep), start_row in bio_rep_starts.items():
                for day in range(9):
                    row_index = start_row + day
                    for condition, cols in condition_cols.items():
                        center = _synthetic_value(day, condition, opt_ref, bio_rep, analyst_index)
                        technical_values = [center - 0.35, center, center + 0.35]
                        sheet.iloc[row_index, cols['tech1']] = technical_values[0]
                        sheet.iloc[row_index, cols['tech2']] = technical_values[1]
                        sheet.iloc[row_index, cols['tech3']] = technical_values[2]
                        sheet.iloc[row_index, cols['mean']] = float(np.mean(technical_values))
                        sheet.iloc[row_index, cols['std']] = float(np.std(technical_values, ddof=1))

            sheet.to_excel(writer, sheet_name=analyst, header=False, index=False)


class PostanalysisSmokeTests(unittest.TestCase):
    """Checks that postanalysis runs on a complete synthetic workbook pair."""

    def setUp(self) -> None:
        self.tmp_root = Path(tempfile.mkdtemp(prefix='autocytotox_postanalysis_'))
        self.results_excel = self.tmp_root / 'autocytotox_results_synthetic.xlsx'
        self.manual_excel = self.tmp_root / 'manual_synthetic.xlsx'
        self.output_dir = self.tmp_root / 'analysis'
        self.output_dir.mkdir()

        _write_results_excel(self.results_excel)
        _write_manual_excel(self.manual_excel)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_run_analysis_writes_expected_outputs(self) -> None:
        from src.postanalysis import run_analysis

        result = run_analysis(
            results_excel=str(self.results_excel),
            manual_excel=str(self.manual_excel),
            output_path=str(self.output_dir),
            export_csv=True,
            plot_format='png',
            verbose=False,
        )

        self.assertFalse(result['auto_agg'].empty)
        self.assertFalse(result['comparison_df'].empty)
        self.assertFalse(result['agreement_metrics_df'].empty)
        self.assertFalse(result['bland_altman_summary_df'].empty)
        self.assertFalse(result['icc_summary_df'].empty)
        self.assertTrue(Path(result['excel_path']).is_file())
        self.assertTrue(Path(result['csv_dir']).is_dir())
        self.assertTrue((self.output_dir / 'analysis_plots').is_dir())


if __name__ == '__main__':
    unittest.main()
