"""
Plotting Module for Flow Cytometry Visualization.

This module provides plotting functions for visualizing flow cytometry data,
including scatter plots, density plots, gating visualizations, and more.
All plotting functions support an optional enable flag for global control.

Author: Aleksander Szarzynski TUW 2026
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class PlotSettings:
    """
    Global settings for plotting behavior.
    
    Attributes
    ----------
    enable_plotting : bool
        Global flag to enable/disable all plotting.
    figure_format : str
        Output format for saved figures ('png', 'pdf', 'svg').
    figure_dpi : int
        DPI for saved figures.
    default_figsize : Tuple[int, int]
        Default figure size (width, height).
    """
    
    enable_plotting: bool = True
    figure_format: str = 'png'
    figure_dpi: int = 150
    default_figsize: Tuple[int, int] = (10, 6)


# Global instance for settings
_settings = PlotSettings()


def get_settings() -> PlotSettings:
    """Get the global PlotSettings instance."""
    return _settings


def set_settings(settings: PlotSettings) -> None:
    """Set the global PlotSettings instance."""
    global _settings
    _settings = settings


def create_folder_if_needed(folder_path: str) -> None:
    """Create folder if it doesn't exist."""
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)


def should_plot(enable: bool = True) -> bool:
    """
    Check if plotting should proceed based on global and local flags.
    
    Parameters
    ----------
    enable : bool, optional
        Local enable flag (default: True).
    
    Returns
    -------
    bool
        True if plotting should proceed.
    """
    return _settings.enable_plotting and enable


def get_save_path(
    save_path: Optional[str],
    subfolder: str,
    filename: str
) -> Optional[str]:
    """
    Generate full save path for a figure.
    
    Parameters
    ----------
    save_path : Optional[str]
        Base save path (None to skip saving).
    subfolder : str
        Subfolder name for organization.
    filename : str
        Figure filename (without extension).
    
    Returns
    -------
    Optional[str]
        Full path to save figure, or None.
    """
    if save_path is None:
        return None
    
    folder = os.path.join(save_path, subfolder)
    create_folder_if_needed(folder)
    return os.path.join(folder, f"{filename}.{_settings.figure_format}")


def plot_scatter(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    title: str = "",
    save_path: Optional[str] = None,
    filename: str = "scatter",
    color: str = 'blue',
    alpha: float = 0.5,
    size: int = 5,
    log_scale: bool = False,
    enable: bool = True,
    **kwargs
) -> None:
    """
    Create a scatter plot of flow cytometry data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    title : str, optional
        Plot title.
    save_path : Optional[str], optional
        Path to save figure.
    filename : str, optional
        Output filename (default: "scatter").
    color : str, optional
        Point color (default: 'blue').
    alpha : float, optional
        Point transparency (default: 0.5).
    size : int, optional
        Point size (default: 5).
    log_scale : bool, optional
        Use log scale for both axes (default: False).
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Examples
    --------
    >>> plot_scatter(data, 'FSC-A', 'SSC-A', title='Cell Population')
    """
    if not should_plot(enable):
        return
    
    plt.figure(figsize=_settings.default_figsize)
    plt.scatter(data[x_axis], data[y_axis], c=color, s=size, alpha=alpha)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, 'scatter', filename)
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()


def plot_beads_clustering(
    beads_data: pd.DataFrame,
    centers: np.ndarray,
    x_axis: str,
    y_axis: str,
    x_cutoff: float,
    n_clusters: int,
    cutoffs: Tuple[float, float, float],
    subfolder: str,
    save_path: Optional[str] = None,
    enable: bool = True
) -> None:
    """
    Plot beads clustering results with gate line.
    
    Parameters
    ----------
    beads_data : pd.DataFrame
        Clustered beads data with 'cluster' column.
    centers : np.ndarray
        Cluster centers from KMeans.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    x_cutoff : float
        Calculated X axis cutoff value.
    n_clusters : int
        Number of clusters used.
    cutoffs : Tuple[float, float, float]
        (x_cutoff_min, x_cutoff_max, y_cutoff_max) used for filtering.
    subfolder : str
        Subfolder name for output.
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Examples
    --------
    >>> plot_beads_clustering(beads, centers, 'FSC-A', 'SSC-A', 1e6, 3, 
    ...                       (0.5e6, 8e6, 16e6), 'day0')
    """
    if not should_plot(enable):
        return
    
    x_cutoff_min, x_cutoff_max, y_cutoff_max = cutoffs
    
    plt.figure(figsize=_settings.default_figsize)
    plt.scatter(
        beads_data[x_axis], beads_data[y_axis],
        c=beads_data['cluster'], s=5, alpha=0.5, cmap='viridis', label='Beads'
    )
    plt.scatter(
        centers[:, 0], centers[:, 1],
        c='red', s=100, marker='X', label='Cluster centers'
    )
    plt.axvline(x=x_cutoff, color='r', linestyle='--', label=f'Gate at {x_cutoff:.2e}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(
        f'KMeans cluster selection with n = {n_clusters}\n'
        f'{x_axis} cutoff min: {x_cutoff_min:.2e}; {x_axis} cutoff max: {x_cutoff_max:.2e}; '
        f'{y_axis} cutoff max: {y_cutoff_max:.2e}'
    )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, 'beads', f'{subfolder}_beads')
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()


def plot_gated_data(
    gated_data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    x_gate_min: float,
    x_gate_max: float,
    y_gate_min: float,
    y_gate_max: float,
    subfolder: str,
    filename: str,
    save_path: Optional[str] = None,
    enable: bool = True
) -> None:
    """
    Plot gated data with gate lines.
    
    Parameters
    ----------
    gated_data : pd.DataFrame
        Gated flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    x_gate_min : float
        X axis minimum gate value.
    x_gate_max : float
        X axis maximum gate value.
    y_gate_min : float
        Y axis minimum gate value.
    y_gate_max : float
        Y axis maximum gate value.
    subfolder : str
        Subfolder identifier.
    filename : str
        Output filename base.
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Examples
    --------
    >>> plot_gated_data(gated, 'FSC-A', 'SSC-A', 1e5, 1e7, 0, 1e7, 
    ...                 'day0', 'sample1')
    """
    if not should_plot(enable):
        return
    
    plt.figure(figsize=_settings.default_figsize)
    plt.scatter(gated_data[x_axis], gated_data[y_axis], c='blue', s=5, label='Gated Data')
    plt.axvline(x=x_gate_min, color='r', linestyle='--', label=f'{x_axis} min: {x_gate_min:.2e}')
    plt.axvline(x=x_gate_max, color='r', linestyle='--', label=f'{x_axis} max: {x_gate_max:.2e}')
    plt.axhline(y=y_gate_min, color='r', linestyle='--', label=f'{y_axis} min: {y_gate_min:.2e}')
    plt.axhline(y=y_gate_max, color='r', linestyle='--', label=f'{y_axis} max: {y_gate_max:.2e}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(
        f'Hardcode gate\n{x_axis} cutoff min: {x_gate_min:.2e}; max: {x_gate_max:.2e}; '
        f'{y_axis} cutoff min: {y_gate_min:.2e}; max: {y_gate_max:.2e}'
    )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, 'Raw_plot_gated', f'{subfolder}_{filename}_gate')
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()


def plot_proximity_knn(
    original_data: pd.DataFrame,
    gated_data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    percent: float,
    plot_data: List[np.ndarray],
    densest_point: Tuple[float, float],
    filename: str,
    save_path: Optional[str] = None,
    enable: bool = True
) -> None:
    """
    Plot proximity-based KNN gating results.
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Original data before gating.
    gated_data : pd.DataFrame
        Data after proximity gating.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    percent : float
        Fraction of data selected.
    plot_data : List[np.ndarray]
        [X, Y, Z] arrays for contour plot.
    densest_point : Tuple[float, float]
        (x, y) coordinates of densest point.
    filename : str
        Output filename base.
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Examples
    --------
    >>> plot_proximity_knn(original, gated, 'SSC-H', 'SSC-A', 0.95, 
    ...                    [X, Y, Z], (1e5, 1e5), 'sample1')
    """
    if not should_plot(enable):
        return
    
    X, Y, Z = plot_data
    
    plt.figure(figsize=_settings.default_figsize)
    plt.scatter(original_data[x_axis], original_data[y_axis], c='lightgrey', s=5, label='All data')
    plt.scatter(gated_data[x_axis], gated_data[y_axis], c='blue', s=5, label=f'{percent*100:.0f}% data')
    plt.contour(X, Y, Z, colors='red')
    plt.scatter(densest_point[0], densest_point[1], c='green', s=100, marker='o', label='Densest Point')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xscale('log')
    plt.yscale('log')
    
    percent_kept = 100 / len(original_data) * len(gated_data) if len(original_data) > 0 else 0
    plt.title(
        f'Proximity selected data with KNN contour {filename}\n'
        f'Original datapoints {len(original_data)} -> new datapoints {len(gated_data)} ({percent_kept:.2f}%)'
    )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, 'proximity_plot', f'{filename}_proximity_knn')
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()


def plot_gate2_scatter(
    original_data: pd.DataFrame,
    gated_data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    gate2_method: str,
    filename: str,
    save_path: Optional[str] = None,
    enable: bool = True
) -> None:
    """Plot gate-2 (singlet/morphology) gating results for ratio or linear_band."""
    if not should_plot(enable):
        return

    plt.figure(figsize=_settings.default_figsize)
    plt.scatter(original_data[x_axis], original_data[y_axis],
                c='lightgrey', s=5, label='All data')
    plt.scatter(gated_data[x_axis], gated_data[y_axis],
                c='blue', s=5, label='Gated')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xscale('log')
    plt.yscale('log')

    n_orig = len(original_data)
    n_gated = len(gated_data)
    pct = 100 * n_gated / n_orig if n_orig > 0 else 0
    plt.title(
        f'Gate 2 ({gate2_method}) – {filename}\n'
        f'{n_orig} → {n_gated} events ({pct:.1f}%)'
    )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if save_path:
        fig_path = get_save_path(save_path, 'gate2_plot',
                                 f'{filename}_gate2_{gate2_method}')
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)

    plt.close()


def plot_kde_with_peaks(
    data: pd.DataFrame,
    channel: str,
    min_value: float,
    peak_x_values: List[float],
    filename: str,
    bw_adjust: float = 0.1,
    save_path: Optional[str] = None,
    enable: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot KDE with identified peaks and local minimum.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    channel : str
        Channel column name.
    min_value : float
        X value of local minimum.
    peak_x_values : List[float]
        X values of identified peaks.
    filename : str
        Output filename base.
    bw_adjust : float, optional
        Bandwidth adjustment for KDE (default: 0.1).
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x_data, y_data) from KDE curve.
    
    Examples
    --------
    >>> x, y = plot_kde_with_peaks(data, 'GFP-A', 1e4, [1e3, 1e5], 'coculture1')
    """
    # Filter positive values
    data_positive = data[data[channel] > 0]
    
    plt.figure(figsize=_settings.default_figsize)
    ax = sns.kdeplot(data_positive[channel], log_scale=True, color='blue', 
                     fill=False, bw_adjust=bw_adjust)
    plt.xscale('log')
    plt.title(f'KDE Plot of {filename}')
    plt.xlabel(channel)
    plt.ylabel('Density')
    
    # Extract KDE data
    line = ax.get_lines()[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    
    if should_plot(enable):
        # Plot peaks and minimum
        for i, peak_x in enumerate(peak_x_values):
            plt.axvline(x=peak_x, color='blue', linestyle='--', 
                       label=f'Peak {i+1}: {peak_x:.2e}')
        plt.axvline(x=min_value, color='red', linestyle='--', 
                   label=f'Local minimum: {min_value:.2e}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        if save_path:
            fig_path = get_save_path(save_path, 'Histogram_CC', f'{filename}_kde')
            if fig_path:
                plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()
    return x_data, y_data


def plot_single_culture_verification(
    effector_data: pd.DataFrame,
    target_data: pd.DataFrame,
    channel: str,
    local_minima: Dict[str, float],
    mean_cutoff: float,
    effector_name: str,
    target_name: str,
    effector_boundaries: Tuple[float, float],
    bw_adjust: float = 0.1,
    save_path: Optional[str] = None,
    enable: bool = True,
    selected_cutoff: Optional[float] = None
) -> None:
    """
    Plot single culture KDE verification with cutoffs.
    
    Parameters
    ----------
    effector_data : pd.DataFrame
        Effector cell data.
    target_data : pd.DataFrame
        Target cell data.
    channel : str
        Channel column name.
    local_minima : Dict[str, float]
        Local minima from cocultures.
    mean_cutoff : float
        Mean cutoff value.
    effector_name : str
        Name of effector cells.
    target_name : str
        Name of target cells.
    effector_boundaries : Tuple[float, float]
        (left_boundary, right_boundary) from KDE threshold.
    bw_adjust : float, optional
        Bandwidth adjustment for KDE (default: 0.1).
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    selected_cutoff : float, optional
        The actual cutoff value used for gating. If provided, drawn as
        a prominent vertical line.
    
    Examples
    --------
    >>> plot_single_culture_verification(nk_data, k562_data, 'GFP-A', 
    ...     {'CC1': 1e4}, 1e4, 'NK', 'K562', (1e3, 1e5))
    """
    if not should_plot(enable):
        return
    
    # Filter positive values
    effector_positive = effector_data[effector_data[channel] > 0]
    target_positive = target_data[target_data[channel] > 0]
    
    plt.figure(figsize=_settings.default_figsize)
    
    # Plot KDEs
    sns.kdeplot(target_positive[channel], log_scale=True, color='green', 
                fill=False, bw_adjust=bw_adjust, label=target_name)
    sns.kdeplot(effector_positive[channel], log_scale=True, color='red', 
                fill=False, bw_adjust=bw_adjust, label=effector_name)
    
    # Plot actual selected cutoff
    if selected_cutoff is not None:
        plt.axvline(x=selected_cutoff, color='red', linestyle='--', linewidth=2,
                   label=f'Selected cutoff: {selected_cutoff:.2e}')
    
    plt.title('KDE Plot - transferred to single cultures')
    plt.xlabel('Intensity (Log Scale)')
    plt.xscale('log')
    plt.ylabel('Density')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, 'Histogram_SC', f'{effector_name}_{target_name}_verification')
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()


def plot_density_histogram(
    density: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    data_name: str,
    original_count: int,
    filtered_count: int,
    x_axis: str,
    y_axis: str,
    log_transform: bool = False,
    bins: int = 300,
    save_path: Optional[str] = None,
    enable: bool = True
) -> None:
    """
    Plot density histogram with cutoff bounds.
    
    Parameters
    ----------
    density : np.ndarray
        Density values.
    lower_bound : float
        Lower cutoff bound.
    upper_bound : float
        Upper cutoff bound.
    data_name : str
        Name of data set.
    original_count : int
        Original number of data points.
    filtered_count : int
        Number of points after filtering.
    x_axis : str
        X axis channel name.
    y_axis : str
        Y axis channel name.
    log_transform : bool, optional
        Whether density was log-transformed (default: False).
    bins : int, optional
        Number of histogram bins (default: 300).
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Examples
    --------
    >>> plot_density_histogram(density, 0.5, 1.5, 'sample1', 10000, 9500, 
    ...                        'SSC-H', 'SSC-A')
    """
    if not should_plot(enable):
        return
    
    transform_label = ' (Log-transformed)' if log_transform else ''
    
    plt.figure(figsize=_settings.default_figsize)
    plt.hist(density, bins=bins, color='green', alpha=0.8, label=f'Density{transform_label}')
    plt.axvline(x=lower_bound, color='red', linestyle='--', label='Mean - 2*SD')
    plt.axvline(x=upper_bound, color='orange', linestyle='--', label='Mean + 2*SD')
    plt.xlabel(f'Density{"log-transformed " if log_transform else ""} - {x_axis}')
    plt.ylabel(f'Frequency - {y_axis}')
    plt.title(
        f'Histogram density distribution of {"log-transformed " if log_transform else ""}{data_name}\n'
        f'Original datapoints {original_count} -> selected datapoints {filtered_count}'
    )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, '2SD_cleaned', f'{data_name}_density')
        if fig_path:
            plt.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close()


def plot_quadrant_gates(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    x_gate: float,
    y_gate: float,
    quadrant_percents: Dict[str, float],
    data_name: str,
    reference_name: str,
    save_path: Optional[str] = None,
    enable: bool = True
) -> None:
    """
    Plot data with quadrant gates and percentages.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to analyze (coculture).
    reference_data : pd.DataFrame
        Reference gating data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    x_gate : float
        X axis gate value.
    y_gate : float
        Y axis gate value.
    quadrant_percents : Dict[str, float]
        Quadrant percentages (q1-q4).
    data_name : str
        Name of data set.
    reference_name : str
        Name of reference set.
    save_path : Optional[str], optional
        Path to save figure.
    enable : bool, optional
        Enable this specific plot (default: True).
    
    Examples
    --------
    >>> plot_quadrant_gates(cc_data, ref_data, 'V450-A', '7-AAD', 
    ...     1e3, 1e3, {'q1': 0.1, 'q2': 0.2, 'q3': 0.5, 'q4': 0.2}, 'CC1', 'NK')
    """
    if not should_plot(enable):
        return
    
    # Filter to positive values for log-scale plotting
    data_pos = data[(data[x_axis] > 0) & (data[y_axis] > 0)]
    ref_pos = reference_data[(reference_data[x_axis] > 0) & (reference_data[y_axis] > 0)]
    
    if data_pos.empty:
        plt.close('all')
        return
    
    fig, ax = plt.subplots(figsize=_settings.default_figsize)
    ax.scatter(data_pos[x_axis], data_pos[y_axis], c='blue', s=5, alpha=0.5, label=data_name)
    if not ref_pos.empty:
        ax.scatter(ref_pos[x_axis], ref_pos[y_axis], c='green', 
                    s=5, alpha=0.5, label=reference_name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Only draw gate lines if values are positive (valid for log scale)
    if x_gate is not None and np.isfinite(x_gate) and x_gate > 0:
        ax.axvline(x=x_gate, color='green', linestyle='--', label=f'Gate at {x_gate:.2e}')
    if y_gate is not None and np.isfinite(y_gate) and y_gate > 0:
        ax.axhline(y=y_gate, color='red', linestyle='--', label=f'Gate at {y_gate:.2e}')
    
    # Quadrant labels
    ax.text(0.87, 1.01, f'Q1: {quadrant_percents["q1"]*100:.2f}%', 
             fontsize=12, transform=ax.transAxes)
    ax.text(0.01, 1.01, f'Q2: {quadrant_percents["q2"]*100:.2f}%', 
             fontsize=12, transform=ax.transAxes)
    ax.text(0.01, 0.02, f'Q3: {quadrant_percents["q3"]*100:.2f}%', 
             fontsize=12, transform=ax.transAxes)
    ax.text(0.87, 0.02, f'Q4: {quadrant_percents["q4"]*100:.2f}%', 
             fontsize=12, transform=ax.transAxes)
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'Hardcoded Gate for {data_name}')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    
    if save_path:
        fig_path = get_save_path(save_path, 'Cytotox_gate', f'{data_name}_distribution')
        if fig_path:
            fig.savefig(fig_path, dpi=_settings.figure_dpi)
    
    plt.close(fig)
