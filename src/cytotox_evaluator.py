"""
Cytotox Evaluator - Main Class for Flow Cytometry Cytotoxicity Analysis.

This module provides the CytotoxEvaluator class for automated gating and
cytotoxicity evaluation of flow cytometry data.

Author: Aleksander Szarzynski TUW 2026
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Python 3.10+ compatibility fix for FlowCytometryTools
import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

from .data_loader import (
    load_fcs_file, get_fcs_files_from_folder, get_beads_file,
    build_identifier_map, create_folder
)
from .analytics import (
    gate_by_threshold, proximity_gate_knn, proximity_gate_knn_euclidean,
    proximity_gate_knn_density, cluster_beads,
    singlet_gate_ratio, singlet_gate_linear_band,
    find_local_minimum_kde,
    calculate_quadrant_percentages, calculate_cytotoxicity,
    derive_gate_value
)
from .plotting import (
    plot_beads_clustering, plot_gated_data,
    plot_proximity_knn, plot_gate2_scatter,
    plot_single_culture_verification,
    plot_quadrant_gates, should_plot,
    get_settings, set_settings
)


@dataclass
class ChannelConfig:
    """
    Configuration for channel names used in analysis.
    
    Attributes
    ----------
    fsc_a : str
        Forward scatter area channel name.
    ssc_a : str
        Side scatter area channel name.
    ssc_h : str
        Side scatter height channel name.
    x_axis_cytotox : str
        Channel name plotted on the cytotoxicity X axis (V450-A / Annexin V).
    y_axis_cytotox : str
        Channel name plotted on the cytotoxicity Y axis (Y690-A / 7-AAD).
    unique_channel : str
        Unique channel for cell discrimination (e.g., GFP).
    """
    fsc_a: str = 'FSC-A'
    ssc_a: str = 'SSC-A'
    ssc_h: str = 'SSC-H'
    x_axis_cytotox: str = 'Annexin V V450-A'
    y_axis_cytotox: str = '7-AAD Y690-A'
    unique_channel: str = 'GFP B525-A'


@dataclass
class GatingConfig:
    """
    Configuration for gating parameters.
    
    Attributes
    ----------
    percent : float
        Percentage of cells to keep in proximity gating (0-1).
    beads_x_cutoff : Optional[float]
        Manual beads cutoff (if None, auto-calculated).
    bw_adjust : float
        Bandwidth adjustment for KDE.
    min_distance : int
        Minimum index distance between KDE peaks.
    subsample : float
        Fraction of events to randomly keep per FCS file (0-1).
        0 or 1 = use all data (no subsampling).
    subsample_seed : int
        Random seed for reproducible subsampling.
    gate2_method : str
        Pre-cytotoxicity singlet/morphology gate on SSC-H vs SSC-A.
    """
    percent: float = 0.95
    beads_x_cutoff: Optional[float] = None
    bw_adjust: float = 0.1
    min_distance: int = 50
    subsample: float = 0.0
    subsample_seed: int = 42
    gate_derivation: str = 'max'
    percentile_q: float = 0.99
    k_sigma: float = 2.5
    gate2_method: str = 'knn'
    gate2_mad_mult: float = 3.0
    gate2_trim_quantile: float = 0.02


@dataclass
class PlotConfig:
    """
    Configuration for plotting options.
    
    Attributes
    ----------
    enable_all : bool
        Global flag to enable/disable all plotting.
    plot_beads : bool
        Enable beads clustering plot.
    plot_gate_lines : bool
        Enable gate line plots.
    plot_proximity : bool
        Enable proximity gating plots.
    plot_kde_minima : bool
        Enable KDE local minima plots.
    plot_sc_verification : bool
        Enable single culture verification plots.
    plot_2sd : bool
        Enable 2SD filtering plots.
    plot_quadrant : bool
        Enable quadrant evaluation plots.
    figure_format : str
        Output figure format.
    """
    enable_all: bool = True
    plot_beads: bool = True
    plot_gate_lines: bool = True
    plot_proximity: bool = True
    plot_kde_minima: bool = True
    plot_sc_verification: bool = True
    plot_2sd: bool = True
    plot_quadrant: bool = True
    figure_format: str = 'png'


class CytotoxEvaluator:
    """
    Main class for automated flow cytometry cytotoxicity evaluation.
    
    This class handles a single folder of FCS files and performs:
    1. Beads-based initial gating
    2. Rectangular threshold gating
    3. Singlet/morphology cleanup gating
    4. Cytotoxicity evaluation using quadrant analysis
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing FCS files.
    identifier_handler : Union[Callable, List[str]]
        Function or list to map filenames to identifiers.
    channels : ChannelConfig, optional
        Channel configuration (default: ChannelConfig()).
    gating : GatingConfig, optional
        Gating configuration (default: GatingConfig()).
    plotting : PlotConfig, optional
        Plotting configuration (default: PlotConfig()).
    output_path : Optional[str], optional
        Path for output files and plots.
    
    Attributes
    ----------
    folder_path : str
        Path to input folder.
    folder_name : str
        Name of input folder.
    data_dict : Dict
        Loaded FCS data organized by identifier.
    processed_dict : Dict
        Processed/gated data.
    beads_cutoff : float
        Calculated beads cutoff value.
    results_df : pd.DataFrame
        Results dataframe with cytotoxicity values.
    
    Examples
    --------
    >>> def get_id(files): return [f.split('_')[1] for f in files]
    >>> evaluator = CytotoxEvaluator('/path/to/data', get_id)
    >>> evaluator.run_full_analysis()
    >>> results = evaluator.get_results()
    
    Notes
    -----
    The expected folder structure contains:
    - One beads file (containing "Beads" in filename)
    - For each experimental group:
      - One effector cell file (containing "NK" by default)
      - One target cell file (containing "K562" by default)
      - One or more coculture files
    """
    
    def __init__(
        self,
        folder_path: str,
        identifier_handler: Union[Callable[[List[str]], List[str]], List[str]],
        channels: ChannelConfig = None,
        gating: GatingConfig = None,
        plotting: PlotConfig = None,
        output_path: Optional[str] = None
    ):
        """Initialize CytotoxEvaluator with folder path and configuration."""
        self.folder_path = folder_path
        self.folder_name = os.path.basename(folder_path)
        self.identifier_handler = identifier_handler
        
        # Configuration
        self.channels = channels or ChannelConfig()
        self.gating = gating or GatingConfig()
        self.plotting = plotting or PlotConfig()
        
        # Set global plotting settings
        settings = get_settings()
        settings.enable_plotting = self.plotting.enable_all
        settings.figure_format = self.plotting.figure_format
        set_settings(settings)
        
        # Output path
        self.output_path = output_path or os.path.join(
            os.path.dirname(folder_path), 'output', self.folder_name
        )
        
        # Data storage
        self.data_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.processed_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.beads_cutoff: float = 0.0
        self.results_df: pd.DataFrame = pd.DataFrame()
        
        # Load data
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Load all FCS files from the folder.
        
        Raises
        ------
        FileNotFoundError
            If folder doesn't exist or no FCS files found.
        ValueError
            If identifier handler produces invalid mapping.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        
        # Get file list (excluding beads)
        fcs_files = get_fcs_files_from_folder(self.folder_path, exclude_pattern="Beads")
        
        if not fcs_files:
            raise FileNotFoundError(f"No FCS files found in: {self.folder_path}")
        
        # Build identifier mapping
        identifier_map = build_identifier_map(fcs_files, self.identifier_handler)
        
        # Subsampling setup
        do_subsample = (0 < self.gating.subsample < 1.0)
        if do_subsample:
            _rng = np.random.RandomState(self.gating.subsample_seed)
        
        # Load each file
        for identifier, files in identifier_map.items():
            self.data_dict[identifier] = {}
            for filename in files:
                file_path = os.path.join(self.folder_path, filename)
                fcs_data = load_fcs_file(file_path, filename)
                
                # Subsample events if requested
                if do_subsample and hasattr(fcs_data, 'data'):
                    n_total = len(fcs_data.data)
                    n_keep = max(1, int(n_total * self.gating.subsample))
                    idx = _rng.choice(n_total, size=n_keep, replace=False)
                    fcs_data.data = fcs_data.data.iloc[idx].reset_index(drop=True)
                
                self.data_dict[identifier][filename] = fcs_data
        
        # Initialize processed dict with same structure
        self.processed_dict = {
            identifier: {filename: None for filename in files}
            for identifier, files in self.data_dict.items()
        }
    
    def setup_beads_gate(
        self,
        x_cutoff_min: float = 0.5e6,
        x_cutoff_max: float = 8e6,
        y_cutoff_max: float = 16e6,
        n_clusters: int = 3,
        n_init: int = 10,
        selected_clusters: Tuple[int, int] = (0, 1)
    ) -> float:
        """
        Setup initial gate based on beads clustering.
        
        Uses KMeans clustering on beads data to determine the FSC-A cutoff
        for separating debris from cells.
        
        Parameters
        ----------
        x_cutoff_min : float, optional
            Minimum X value for beads selection (default: 0.5e6).
        x_cutoff_max : float, optional
            Maximum X value for beads selection (default: 8e6).
        y_cutoff_max : float, optional
            Maximum Y value for beads selection (default: 16e6).
        n_clusters : int, optional
            Number of KMeans clusters (default: 3).
        n_init : int, optional
            Number of KMeans initializations (default: 10).
        selected_clusters : Tuple[int, int], optional
            Cluster indices for cutoff calculation (default: (0, 1)).
        
        Returns
        -------
        float
            Calculated X-axis cutoff value.
        
        Raises
        ------
        ValueError
            If beads file not found or multiple found.
        
        Examples
        --------
        >>> cutoff = evaluator.setup_beads_gate(n_clusters=4, selected_clusters=(1, 2))
        """
        if self.gating.beads_x_cutoff is not None:
            self.beads_cutoff = self.gating.beads_x_cutoff
            return self.beads_cutoff

        # OPT-IN auto-derivation path: only valid if size-calibration beads
        # were actually acquired on the plate. The cutoff can drift when the
        # bead signal is noisy -- use at your own risk.
        warnings.warn(
            'beads_x_cutoff is None: deriving the FSC-A cutoff from a Size '
            'Beads .fcs file. This mode is OPTIONAL and only valid if size-'
            'calibration beads were actually measured on the plate. Noisy '
            'bead signals can shift the auto-derived cutoff -- use at your '
            'own risk. Set gating.beads_x_cutoff to a fixed FSC-A value to '
            'restore the recommended hard-coded gate.',
            stacklevel=2,
        )

        beads_filename = get_beads_file(self.folder_path)
        if beads_filename is None:
            raise ValueError(f"No beads file found in {self.folder_path}")
        
        beads_path = os.path.join(self.folder_path, beads_filename)
        beads_fcs = load_fcs_file(beads_path, f'beads_{self.folder_name}')
        beads_data = beads_fcs.data[[self.channels.fsc_a, self.channels.ssc_a]]
        
        # Cluster beads
        clustered_data, x_cutoff, centers = cluster_beads(
            beads_data,
            self.channels.fsc_a,
            self.channels.ssc_a,
            x_cutoff_min=x_cutoff_min,
            x_cutoff_max=x_cutoff_max,
            y_cutoff_max=y_cutoff_max,
            n_clusters=n_clusters,
            n_init=n_init,
            selected_clusters=selected_clusters
        )
        
        self.beads_cutoff = x_cutoff
        
        # Plot if enabled
        if should_plot(self.plotting.plot_beads):
            create_folder(self.output_path)
            plot_beads_clustering(
                clustered_data, centers,
                self.channels.fsc_a, self.channels.ssc_a,
                x_cutoff, n_clusters,
                (x_cutoff_min, x_cutoff_max, y_cutoff_max),
                self.folder_name,
                save_path=self.output_path,
                enable=True
            )
        
        return self.beads_cutoff
    
    def apply_threshold_gate(
        self,
        x_min: Optional[float] = None,
        x_max: float = 1e7,
        y_min: float = 0,
        y_max: float = 1e7,
        data_dict: Optional[Dict] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply rectangular threshold gating to data.
        
        Parameters
        ----------
        x_min : Optional[float], optional
            Minimum X threshold (default: beads_cutoff).
        x_max : float, optional
            Maximum X threshold (default: 1e7).
        y_min : float, optional
            Minimum Y threshold (default: 0).
        y_max : float, optional
            Maximum Y threshold (default: 1e7).
        data_dict : Optional[Dict], optional
            Data to gate (default: self.data_dict).
        
        Returns
        -------
        Dict
            Gated data dictionary.
        
        Examples
        --------
        >>> gated = evaluator.apply_threshold_gate(x_min=1e5)
        """
        if x_min is None:
            x_min = self.beads_cutoff
        
        if data_dict is None:
            data_dict = self.data_dict
        
        gated_dict = {}
        
        for identifier, files in data_dict.items():
            gated_dict[identifier] = {}
            
            for filename, fcs_data in files.items():
                # Get raw data from FCMeasurement or DataFrame
                if hasattr(fcs_data, 'data'):
                    raw_data = fcs_data.data
                else:
                    raw_data = fcs_data
                
                gated_data = gate_by_threshold(
                    raw_data,
                    self.channels.fsc_a,
                    self.channels.ssc_a,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
                )
                
                gated_dict[identifier][filename] = gated_data
                
                # Plot if enabled
                if should_plot(self.plotting.plot_gate_lines):
                    create_folder(self.output_path)
                    filename_clean = filename.split('.')[0]
                    plot_gated_data(
                        gated_data,
                        self.channels.fsc_a,
                        self.channels.ssc_a,
                        x_min, x_max, y_min, y_max,
                        self.folder_name,
                        filename_clean,
                        save_path=os.path.join(self.output_path, 'gate1'),
                        enable=True
                    )
        
        self.processed_dict = gated_dict
        return gated_dict
    
    def apply_proximity_gate(
        self,
        data_dict: Optional[Dict] = None,
        percent: Optional[float] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply the configurable gate-2 singlet/morphology gate.
        
        Parameters
        ----------
        data_dict : Optional[Dict], optional
            Data to gate (default: self.processed_dict).
        percent : Optional[float], optional
            Fraction of data to keep for the legacy knn method.
        
        Returns
        -------
        Dict
            Proximity-gated data dictionary.
        
        Examples
        --------
        >>> gated = evaluator.apply_proximity_gate(percent=0.9)
        """
        if data_dict is None:
            data_dict = self.processed_dict
        
        if percent is None:
            percent = self.gating.percent
        
        gated_dict = {}
        
        for identifier, files in data_dict.items():
            gated_dict[identifier] = {}
            
            for filename, fcs_data in files.items():
                gate2_method = getattr(self.gating, 'gate2_method', 'knn')

                if gate2_method == 'knn' and self.plotting.plot_proximity and should_plot():
                    gated_data, plot_data, densest = proximity_gate_knn(
                        fcs_data,
                        self.channels.ssc_h,
                        self.channels.ssc_a,
                        percent=percent,
                        return_plot_data=True
                    )
                    
                    create_folder(self.output_path)
                    filename_clean = filename.split('.')[0]
                    plot_proximity_knn(
                        fcs_data, gated_data,
                        self.channels.ssc_h, self.channels.ssc_a,
                        percent, plot_data, densest,
                        filename_clean,
                        save_path=os.path.join(self.output_path, 'gate2'),
                        enable=True
                    )
                else:
                    if gate2_method == 'ratio':
                        gated_data = singlet_gate_ratio(
                            fcs_data,
                            self.channels.ssc_h,
                            self.channels.ssc_a,
                            mad_mult=getattr(self.gating, 'gate2_mad_mult', 3.0)
                        )
                    elif gate2_method == 'linear_band':
                        gated_data = singlet_gate_linear_band(
                            fcs_data,
                            self.channels.ssc_h,
                            self.channels.ssc_a,
                            mad_mult=getattr(self.gating, 'gate2_mad_mult', 3.0),
                            trim_quantile=getattr(self.gating, 'gate2_trim_quantile', 0.02)
                        )
                    else:
                        gated_data = proximity_gate_knn(
                            fcs_data,
                            self.channels.ssc_h,
                            self.channels.ssc_a,
                            percent=percent,
                            return_plot_data=False
                        )

                    # Plot gate2 for non-knn methods
                    if self.plotting.plot_proximity and should_plot():
                        create_folder(self.output_path)
                        filename_clean = filename.split('.')[0]
                        plot_gate2_scatter(
                            fcs_data, gated_data,
                            self.channels.ssc_h, self.channels.ssc_a,
                            gate2_method, filename_clean,
                            save_path=os.path.join(self.output_path, 'gate2'),
                            enable=True
                        )
                
                gated_dict[identifier][filename] = gated_data
        
        self.processed_dict = gated_dict
        return gated_dict
    
    def evaluate_cytotoxicity(
        self,
        data_dict: Optional[Dict] = None,
        effector_cell: str = 'NK',
        target_cell: str = 'K562',
        proximity_method: str = 'axis'
    ) -> pd.DataFrame:
        """
        Evaluate cytotoxicity using quadrant analysis.
        
        This method:
        1. Identifies local minima in coculture KDE plots
        2. Applies cutoffs to separate effector and target cells
        3. Calculates quadrant percentages for cytotoxicity
        4. Computes final cytotoxicity values
        
        Parameters
        ----------
        data_dict : Optional[Dict], optional
            Data to analyze (default: self.processed_dict).
        effector_cell : str, optional
            Pattern to identify effector cells (default: 'NK').
        target_cell : str, optional
            Pattern to identify target cells (default: 'K562').
        proximity_method : str, optional
            Method for proximity gating: 'axis' (default), 'euclidean', or 'density'.
        
        Returns
        -------
        pd.DataFrame
            Results dataframe with cytotoxicity values.
        
        Raises
        ------
        ValueError
            If expected file structure is not found.
        
        Examples
        --------
        >>> results = evaluator.evaluate_cytotoxicity()
        >>> print(results[['File', 'Cytotoxicity']])
        """
        if data_dict is None:
            data_dict = self.processed_dict
        
        results_list = []
        
        for identifier, files in data_dict.items():
            wells = list(files.keys())
            
            # Separate effector, target, and coculture files
            effector_dict = {w: files[w] for w in wells if effector_cell in w}
            target_dict = {w: files[w] for w in wells if target_cell in w}
            coculture_dict = {w: files[w] for w in wells 
                            if effector_cell not in w and target_cell not in w}
            
            # Validate structure
            if len(effector_dict) != 1 or len(target_dict) != 1 or len(coculture_dict) < 1:
                raise ValueError(
                    f"Invalid file structure in {identifier}: "
                    f"effector={len(effector_dict)}, target={len(target_dict)}, "
                    f"coculture={len(coculture_dict)}"
                )
            
            # Get local minima from cocultures
            local_minima, cc_kde_data = self._get_local_minima(coculture_dict)
            
            # Apply minima cutoffs
            effector_gated, target_gated, cutoff = self._apply_local_minima(
                local_minima, effector_dict, target_dict
            )
            
            # Plot Histogram_CC with the selected cutoff
            self._plot_histogram_cc(cc_kde_data, cutoff)
            
            # Select proximity gate function based on method
            if proximity_method == 'euclidean':
                proximity_func = proximity_gate_knn_euclidean
            elif proximity_method == 'density':
                proximity_func = proximity_gate_knn_density
            else:  # 'axis' or default
                proximity_func = proximity_gate_knn
            
            # Proximity gate for final analysis
            effector_final = proximity_func(
                list(effector_gated.values())[0],
                self.channels.x_axis_cytotox,
                self.channels.y_axis_cytotox,
                percent=self.gating.percent,
                return_plot_data=False
            )
            
            target_final = proximity_func(
                list(target_gated.values())[0],
                self.channels.x_axis_cytotox,
                self.channels.y_axis_cytotox,
                percent=self.gating.percent,
                return_plot_data=False
            )
            
            # Calculate quadrant percentages for all samples
            identifier_results = self._calculate_quadrants(
                coculture_dict, effector_dict, target_dict,
                effector_final, target_final,
                cutoff, effector_cell, target_cell
            )
            
            # Calculate cytotoxicity
            identifier_results = self._calculate_cytotox_values(
                identifier_results, effector_cell, target_cell
            )
            
            results_list.append(identifier_results)
        
        self.results_df = pd.concat(results_list, ignore_index=True)
        self.results_df.insert(0, 'Folder', self.folder_name)
        
        return self.results_df
    
    def _compute_kde_curve(
        self,
        values: pd.Series,
        n_points: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        positive = values[values > 0].to_numpy(dtype=float)
        if positive.size < 2:
            raise ValueError('At least two positive values are required for KDE estimation')

        log_values = np.log10(positive)
        hist, bin_edges = np.histogram(log_values, bins=n_points, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        sigma_bins = max(1.0, float(self.gating.bw_adjust) * 12.0)
        radius = max(3, int(np.ceil(4 * sigma_bins)))
        kernel_positions = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (kernel_positions / sigma_bins) ** 2)
        kernel /= kernel.sum()

        y_data = np.convolve(hist, kernel, mode='same')
        x_data = np.power(10.0, bin_centers)
        return x_data, y_data

    def _get_local_minima(self, coculture_dict: Dict) -> Tuple[Dict, Dict]:
        """Get local minima from coculture KDE plots.

        Returns
        -------
        local_minima : dict
            {cc_name: min_value}
        cc_kde_data : dict
            {cc_name: (data, min_value)} stored for deferred plotting.
        """
        local_minima = {}
        cc_kde_data = {}
        
        for cc_name, coculture in coculture_dict.items():
            # Filter positive values
            data = coculture[coculture[self.channels.unique_channel] > 0]
            try:
                x_data, y_data = self._compute_kde_curve(data[self.channels.unique_channel])
                min_value, _highest_peaks = find_local_minimum_kde(
                    x_data, y_data, self.gating.min_distance
                )
            except Exception as exc:
                raise ValueError(
                    f'Could not derive GFP separation cutoff for {cc_name!r} '
                    f'in folder {self.folder_name!r} using channel '
                    f'{self.channels.unique_channel!r}. Check the FCS signal, '
                    f'bw_adjust={self.gating.bw_adjust}, and '
                    f'min_distance={self.gating.min_distance}.'
                ) from exc
            
            local_minima[cc_name] = min_value
            cc_kde_data[cc_name] = (data, min_value)
        
        return local_minima, cc_kde_data

    def _plot_histogram_cc(self, cc_kde_data: Dict, selected_cutoff: float):
        """Plot Histogram_CC for each coculture with the selected cutoff."""
        if not should_plot(self.plotting.plot_kde_minima):
            return

        settings = get_settings()
        save_folder = os.path.join(
            self.output_path, 'cytotoxicity_evaluator', 'Histogram_CC')
        create_folder(save_folder)

        for cc_name, (data, min_value) in cc_kde_data.items():
            cc_name_clean = str(cc_name).split('.')[0]
            x_data, y_data = self._compute_kde_curve(data[self.channels.unique_channel])

            plt.figure(figsize=(7, 4))
            plt.plot(x_data, y_data, color='blue')
            plt.axvline(x=selected_cutoff, color='red', linestyle='--',
                       linewidth=2,
                       label=f'Selected cutoff: {selected_cutoff:.2e}')
            plt.xscale('log')
            plt.title(f'KDE Plot of {cc_name_clean}')
            plt.xlabel(self.channels.unique_channel)
            plt.ylabel('Density')
            plt.legend(loc='upper right')
            plt.tight_layout()

            fig_path = os.path.join(
                save_folder, f'{cc_name}_kde.{settings.figure_format}')
            plt.savefig(fig_path, dpi=settings.figure_dpi)
            plt.close()
    
    def _apply_local_minima(
        self,
        local_minima: Dict[str, float],
        effector_dict: Dict,
        target_dict: Dict
    ) -> Tuple[Dict, Dict, float]:
        """Apply local minima cutoffs to single cultures."""
        effector_name = next(iter(effector_dict)).split('.')[0]
        target_name = next(iter(target_dict)).split('.')[0]
        
        effector_data = next(iter(effector_dict.values()))
        target_data = next(iter(target_dict.values()))
        
        # Mean local minimum
        mean_minimum = np.mean(list(local_minima.values()))
        
        # Filter positive values
        effector_data = effector_data[effector_data[self.channels.unique_channel] > 0]
        target_data = target_data[target_data[self.channels.unique_channel] > 0]

        # Get KDE boundaries for effector
        x_data, y_data = self._compute_kde_curve(effector_data[self.channels.unique_channel])
        
        peak_idx = np.argmax(y_data)
        peak_value = y_data[peak_idx]
        threshold = 0.05 * peak_value

        left_candidates = np.where(y_data[:peak_idx] < threshold)[0]
        right_candidates = np.where(y_data[peak_idx:] < threshold)[0]
        if left_candidates.size == 0 or right_candidates.size == 0:
            raise ValueError(
                f'Could not identify effector KDE peak boundaries in folder '
                f'{self.folder_name!r} for {effector_name!r} using channel '
                f'{self.channels.unique_channel!r}. Check the single-culture '
                f'signal or adjust bw_adjust={self.gating.bw_adjust}.'
            )

        left_idx = left_candidates[-1]
        right_idx = right_candidates[0] + peak_idx
        
        cutoff = x_data[right_idx]
        
        # Plot verification if enabled
        if should_plot(self.plotting.plot_sc_verification):
            plot_single_culture_verification(
                effector_data, target_data,
                self.channels.unique_channel,
                local_minima, mean_minimum,
                effector_name, target_name,
                (x_data[left_idx], cutoff),
                bw_adjust=self.gating.bw_adjust,
                save_path=os.path.join(self.output_path, 'cytotoxicity_evaluator'),
                enable=True,
                selected_cutoff=cutoff,
            )
        
        # Apply gating
        effector_gated = {
            effector_name: effector_data[effector_data[self.channels.unique_channel] < cutoff]
        }
        target_gated = {
            target_name: target_data[target_data[self.channels.unique_channel] > cutoff]
        }
        
        return effector_gated, target_gated, cutoff
    
    def _calculate_quadrants(
        self,
        coculture_dict: Dict,
        effector_dict: Dict,
        target_dict: Dict,
        effector_ref: pd.DataFrame,
        target_ref: pd.DataFrame,
        cutoff: float,
        effector_cell: str,
        target_cell: str
    ) -> pd.DataFrame:
        """Calculate quadrant percentages for all samples."""
        results = []

        gd = self.gating.gate_derivation
        pq = self.gating.percentile_q
        ks = self.gating.k_sigma

        x_gate_eff = derive_gate_value(effector_ref[self.channels.x_axis_cytotox], gd, pq, ks)
        y_gate_eff = derive_gate_value(effector_ref[self.channels.y_axis_cytotox], gd, pq, ks)
        x_gate_tgt = derive_gate_value(target_ref[self.channels.x_axis_cytotox], gd, pq, ks)
        y_gate_tgt = derive_gate_value(target_ref[self.channels.y_axis_cytotox], gd, pq, ks)
        
        # Process effector gating (cocultures + effector)
        for culture_name, culture_data in {**coculture_dict, **effector_dict}.items():
            is_effector_sc = culture_name in effector_dict
            if is_effector_sc:
                # Effector single culture is GFP-negative → keep events below cutoff
                gated = culture_data[culture_data[self.channels.unique_channel] < cutoff]
            else:
                # Coculture → keep GFP+ events above cutoff
                gated = culture_data[culture_data[self.channels.unique_channel] > cutoff]
            if gated.empty:
                gated = culture_data[culture_data[self.channels.unique_channel] == 
                                     culture_data[self.channels.unique_channel].max()]
            
            percents = calculate_quadrant_percentages(
                gated, self.channels.x_axis_cytotox, self.channels.y_axis_cytotox,
                x_gate_eff, y_gate_eff
            )
            
            row = {'File': culture_name}
            for q, p in percents.items():
                row[f'{q}_{effector_cell}'] = p
            results.append(row)
            
            # Plot if enabled
            if should_plot(self.plotting.plot_quadrant):
                culture_clean = culture_name.split('.')[0]
                plot_quadrant_gates(
                    gated, effector_ref,
                    self.channels.x_axis_cytotox, self.channels.y_axis_cytotox,
                    x_gate_eff, y_gate_eff,
                    percents, culture_clean, effector_cell,
                    save_path=os.path.join(self.output_path, 'cytotoxicity_evaluator'),
                    enable=True
                )
        
        df = pd.DataFrame(results)
        
        # Process target gating (cocultures + target)
        for culture_name, culture_data in {**coculture_dict, **target_dict}.items():
            # Both cocultures and target single culture use > cutoff (GFP+ events)
            gated = culture_data[culture_data[self.channels.unique_channel] > cutoff]
            
            percents = calculate_quadrant_percentages(
                gated, self.channels.x_axis_cytotox, self.channels.y_axis_cytotox,
                x_gate_tgt, y_gate_tgt
            )
            
            for q, p in percents.items():
                df.loc[df['File'] == culture_name, f'{q}_{target_cell}'] = p
            
            # Add row if not exists (target only)
            if culture_name not in df['File'].values:
                row = {'File': culture_name}
                for q, p in percents.items():
                    row[f'{q}_{target_cell}'] = p
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
            # Plot if enabled
            if should_plot(self.plotting.plot_quadrant):
                culture_clean = culture_name.split('.')[0]
                plot_quadrant_gates(
                    gated, target_ref,
                    self.channels.x_axis_cytotox, self.channels.y_axis_cytotox,
                    x_gate_tgt, y_gate_tgt,
                    percents, culture_clean, target_cell,
                    save_path=os.path.join(self.output_path, 'cytotoxicity_evaluator'),
                    enable=True
                )
        
        return df
    
    def _calculate_cytotox_values(
        self,
        df: pd.DataFrame,
        effector_cell: str,
        target_cell: str
    ) -> pd.DataFrame:
        """Calculate final cytotoxicity values."""
        # Find target baseline (single culture)
        target_rows = df[df['File'].str.contains(target_cell)]
        if target_rows.empty:
            return df
        
        target_baseline = target_rows[f'q3_{target_cell}'].iloc[0]
        
        # Find coculture rows (not effector or target)
        coculture_mask = ~df['File'].str.contains(target_cell) & ~df['File'].str.contains(effector_cell)
        
        for idx in df[coculture_mask].index:
            live_percent = df.loc[idx, f'q3_{target_cell}']
            
            if pd.notna(live_percent):
                # Calculate cytotoxicity components
                df.loc[idx, f'{target_cell}_Live'] = 100 - ((target_baseline - live_percent) * 100)
                df.loc[idx, f'{target_cell}_Late_Apoptotic'] = (df.loc[idx, f'q1_{target_cell}'] - target_rows[f'q1_{target_cell}'].iloc[0]) * 100
                df.loc[idx, f'{target_cell}_Necrotic'] = (df.loc[idx, f'q2_{target_cell}'] - target_rows[f'q2_{target_cell}'].iloc[0]) * 100
                df.loc[idx, f'{target_cell}_Early_Apoptotic'] = (df.loc[idx, f'q4_{target_cell}'] - target_rows[f'q4_{target_cell}'].iloc[0]) * 100
                
                # Calculate cytotoxicity
                cytotox = calculate_cytotoxicity(
                    df.loc[idx, f'{target_cell}_Live'],
                    target_baseline,
                    to_zero=True
                )
                df.loc[idx, 'Cytotoxicity'] = cytotox
        
        # Calculate mean and std for cocultures
        coculture_cytotox = df.loc[coculture_mask, 'Cytotoxicity'].dropna()
        if not coculture_cytotox.empty:
            first_cc_idx = df[coculture_mask].index[0]
            df.loc[first_cc_idx, 'Cytotoxicity_mean'] = coculture_cytotox.mean()
            df.loc[first_cc_idx, 'Cytotoxicity_std'] = coculture_cytotox.std()
        
        return df
    
    def run_full_analysis(
        self,
        beads_params: Optional[Dict] = None,
        gate_params: Optional[Dict] = None,
        proximity_params: Optional[Dict] = None,
        cytotox_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Run complete analysis pipeline.
        
        Executes all analysis steps in sequence:
        1. Beads gate setup
        2. Threshold gating
        3. Proximity gating
        4. Cytotoxicity evaluation
        
        Parameters
        ----------
        beads_params : Optional[Dict], optional
            Parameters for setup_beads_gate().
        gate_params : Optional[Dict], optional
            Parameters for apply_threshold_gate().
        proximity_params : Optional[Dict], optional
            Parameters for apply_proximity_gate().
        cytotox_params : Optional[Dict], optional
            Parameters for evaluate_cytotoxicity().
        
        Returns
        -------
        pd.DataFrame
            Complete results dataframe.
        
        Examples
        --------
        >>> results = evaluator.run_full_analysis(
        ...     beads_params={'n_clusters': 4},
        ...     cytotox_params={'effector_cell': 'NK92'}
        ... )
        """
        beads_params = beads_params or {}
        gate_params = gate_params or {}
        proximity_params = proximity_params or {}
        cytotox_params = cytotox_params or {}
        
        # Step 1: Setup beads gate
        self.setup_beads_gate(**beads_params)
        
        # Step 2: Apply threshold gate
        self.apply_threshold_gate(**gate_params)
        
        # Step 3: Apply proximity gate
        self.apply_proximity_gate(**proximity_params)
        
        # Step 4: Evaluate cytotoxicity
        results = self.evaluate_cytotoxicity(**cytotox_params)
        
        return results
    
    def get_results(self) -> pd.DataFrame:
        """
        Get current results dataframe.
        
        Returns
        -------
        pd.DataFrame
            Results dataframe with cytotoxicity values.
        """
        return self.results_df.copy()
    
    def save_results(
        self,
        output_path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save results to Excel file.
        
        Parameters
        ----------
        output_path : Optional[str], optional
            Output directory (default: self.output_path).
        filename : Optional[str], optional
            Output filename (auto-generated if None).
        
        Returns
        -------
        str
            Full path to saved file.
        
        Examples
        --------
        >>> path = evaluator.save_results()
        >>> print(f"Results saved to: {path}")
        """
        if output_path is None:
            output_path = self.output_path
        
        create_folder(output_path)
        
        if filename is None:
            timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            filename = f'{self.folder_name}_{timestamp}_{self.gating.percent}.xlsx'
        
        full_path = os.path.join(output_path, filename)
        self.results_df.to_excel(full_path, index=False)
        
        return full_path
