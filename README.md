# AutoCytotox

AutoCytotox is a deterministic autogating workflow for flow-cytometric NK-cell
cytotoxicity analysis against K562 target cells. It standardizes event
selection, target-effector separation, apoptosis-space thresholding, replicate
aggregation, and optional comparison against manual evaluation.

The repository contains a bundled example dataset, a YAML-driven main workflow,
a grid-search workflow, and manuscript-aligned postanalysis tools.

## Documentation

This README is the main user-facing guide for installation, configuration,
commands, outputs, tests, and troubleshooting.

- [examples/simple_example.ipynb](examples/simple_example.ipynb): notebook
  walkthrough using the bundled example data.

## Requirements

- Windows, Linux, or macOS
- Conda
- Python 3.10 environment with the packages in [environment.yml](environment.yml)
- FCS files according to data layout 

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate autogating_env
```

## Quick Start

Run the bundled example:

```bash
python autocytotox.py --config examples/simple_example_config.yaml
```

Preview the grid-search smoke configuration:

```bash
python autocytotox_gridsearch.py --config gridsearch_config.yaml --dry-run
```

Run the main workflow with a manual-reference workbook and manuscript-aligned
postanalysis:

```bash
python autocytotox.py --config my_config.yaml --manual-excel manual_reference.xlsx --run-analysis --export-csv
```

## Data Layout

AutoCytotox discovers subfolders whose names start with `Day `:

```text
example_data/
  Day 0_B/
    03-CC_B_Opt_01-D1.fcs
    03-CC_B_Opt_01-D2.fcs
    03-CC_B_Opt_01-D3.fcs
    03-NK_B_Opt_only-D4.fcs
    03-K562_only-DX.fcs
    ...
  Day 1_B/
    *.fcs
```

Folder suffixes encode culture condition labels:

- `B`: batch
- `F`: fed batch
- `R`: repetitive batch

Filenames also encode the media or process arm:

- `Opt`: optimized condition
- `Ref`: reference condition

For example, `03-CC_B_Opt_01-D1.fcs` means a coculture file from condition
`B`, arm `Opt`, biological replicate `01`, well `D1`.

## File Grouping

Within each `Day <N>_<condition>` folder, AutoCytotox clusters files into
matched analysis groups before cytotoxicity is calculated. The default
`workflow.identifier_strategy: well_row` uses the row letter from the trailing
well token:

```text
03-CC_B_Opt_01-D1.fcs       -> group D
03-CC_B_Opt_01-D2.fcs       -> group D
03-CC_B_Opt_01-D3.fcs       -> group D
03-NK_B_Opt_only-D4.fcs     -> group D
03-K562_only-DX.fcs         -> group D
```

That group is then analyzed as one matched set. The same pattern applies to
reference groups, for example row `A`, `B`, or `C` for `Ref` files, and to
other condition folders such as `Day 2_R`.

Each matched group should contain:

- one effector-only file, by default labelled `NK`
- one target-only file, by default labelled `K562`
- one or more coculture files

The built-in grouping strategies are:

- `well_row`: uses only the row letter from the trailing well token, such as
  `D` from `D1`. This is the bundled-data default.
- `well_position`: uses the full trailing well token, such as `D1`.
- `stem`: uses the full filename without `.fcs`.

If your filenames use a different convention, set
`workflow.identifier_callable` to a custom `module:function` or
`path/to/file.py:function` that returns one group label per filename.

## Configuration

For a new dataset, copy [examples/simple_example_config.yaml](examples/simple_example_config.yaml)
or [config.yaml](config.yaml), then edit:

- `paths.data_path`
- `paths.output_path`
- `channels.*` if your FCS channel labels differ
- `workflow.identifier_strategy` or `workflow.identifier_callable` if your
  filenames use a different grouping scheme

The validated default workflow is (check yaml schema for details):

```yaml
workflow:
  method: euclidean
gating:
  percent: 0.95
  bw_adjust: 0.05
  min_distance: 30
  gate2_method: ratio
  gate_derivation: percentile
  percentile_q: 0.90
  beads_x_cutoff: 2400000
```

`beads_x_cutoff: 2400000` is the fixed FSC-A threshold (adjust as needed). 
Set it to `null` in YAML, or pass `--beads-x-cutoff 0`, only when Size Beads were
measured and auto-derivation is intended (CAUTION - use with care).

## Outputs

The main workflow writes a timestamped Excel workbook with raw per-file
results, aggregated cytotoxicity summaries, run parameters, and optional manual
comparison metrics. When plotting is enabled, diagnostic figures are written
under the output folder by day and analysis stage.

With `--run-analysis --export-csv`, postanalysis can also export retained
tables and plots for:

- pooled manual comparison metrics
- ICC(3,1)
- leave-one-out Bland-Altman bias
- mixed-effects variance decomposition
- mixed-model residual normality diagnostics
- cytotoxicity-vs-CV Spearman and Wilcoxon breakpoint analysis

## Project Layout

| Path | Purpose |
|---|---|
| `autocytotox.py` | Main YAML-driven workflow entry point |
| `autocytotox_gridsearch.py` | Grid-search entry point |
| `src/workflow/` | Run orchestration, folder processing, aggregation, and manual comparison helpers |
| `src/gridsearch/` | Grid expansion, runner config construction, resume handling, and execution |
| `src/postanalysis/` | Manuscript-aligned postanalysis and cytotoxicity-vs-CV precision analysis |
| `src/` | Shared loading, gating, identifiers, plotting, reporting, and compatibility utilities |
| `example_data/` | Bundled FCS subset for smoke testing |
| `examples/` | Example configuration and notebook walkthrough |
| `tests/` | Standard-library smoke and configuration tests |

## Citation And License

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The
manuscript citation block is a placeholder until final publication metadata are
available.

This repository is distributed under the MIT license. See [LICENSE](LICENSE).
