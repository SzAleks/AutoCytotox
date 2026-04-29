"""
Analytics Module for Flow Cytometry Data Analysis.

This module provides analytical functions for processing flow cytometry data,
including gating operations, density estimation, and cytotoxicity calculations.

Author: Aleksander Szarzynski TUW 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def gate_by_threshold(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    x_min: float = 0,
    x_max: float = 1e7,
    y_min: float = 0,
    y_max: float = 1e7
) -> pd.DataFrame:
    """
    Apply rectangular threshold gating to flow cytometry data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data with channel columns.
    x_axis : str
        Column name for X axis gating.
    y_axis : str
        Column name for Y axis gating.
    x_min : float, optional
        Minimum X threshold (default: 0).
    x_max : float, optional
        Maximum X threshold (default: 1e7).
    y_min : float, optional
        Minimum Y threshold (default: 0).
    y_max : float, optional
        Maximum Y threshold (default: 1e7).
    
    Returns
    -------
    pd.DataFrame
        Gated data subset.
    
    Examples
    --------
    >>> gated = gate_by_threshold(data, 'FSC-A', 'SSC-A', x_min=1e5, x_max=1e6)
    """
    mask = (
        (data[x_axis] >= x_min) & 
        (data[x_axis] <= x_max) &
        (data[y_axis] >= y_min) & 
        (data[y_axis] <= y_max)
    )
    return data[mask].copy()


def singlet_gate_ratio(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    mad_mult: float = 3.0,
    small_num: float = 1e-6
) -> pd.DataFrame:
    """Gate singlet-like events using a robust log-ratio filter on y/x."""
    xvals = data[x_axis].to_numpy(dtype=float)
    yvals = data[y_axis].to_numpy(dtype=float)

    valid = np.isfinite(xvals) & np.isfinite(yvals) & (xvals > 0) & (yvals > 0)
    if valid.sum() < 10:
        return data.loc[valid].copy()

    ratio_log = np.log10(np.maximum(yvals[valid], small_num) /
                         np.maximum(xvals[valid], small_num))
    center = np.median(ratio_log)
    mad = np.median(np.abs(ratio_log - center))
    if mad <= 0:
        mad = np.std(ratio_log)
    if mad <= 0:
        return data.loc[valid].copy()

    keep_valid = np.abs(ratio_log - center) <= (mad_mult * mad)
    keep_mask = np.zeros(len(data), dtype=bool)
    keep_mask[np.where(valid)[0][keep_valid]] = True
    return data.loc[keep_mask].copy()


def singlet_gate_linear_band(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    mad_mult: float = 3.0,
    trim_quantile: float = 0.02
) -> pd.DataFrame:
    """Gate singlet-like events using a robust linear band on y vs x."""
    xvals = data[x_axis].to_numpy(dtype=float)
    yvals = data[y_axis].to_numpy(dtype=float)

    valid = np.isfinite(xvals) & np.isfinite(yvals)
    if valid.sum() < 10:
        return data.loc[valid].copy()

    x_valid = xvals[valid]
    y_valid = yvals[valid]
    if 0 < trim_quantile < 0.25:
        x_lo, x_hi = np.quantile(x_valid, [trim_quantile, 1 - trim_quantile])
        y_lo, y_hi = np.quantile(y_valid, [trim_quantile, 1 - trim_quantile])
        central = (
            (x_valid >= x_lo) & (x_valid <= x_hi) &
            (y_valid >= y_lo) & (y_valid <= y_hi)
        )
    else:
        central = np.ones_like(x_valid, dtype=bool)

    if central.sum() < 10:
        central = np.ones_like(x_valid, dtype=bool)

    slope, intercept = np.polyfit(x_valid[central], y_valid[central], 1)
    residuals = y_valid - (slope * x_valid + intercept)
    center = np.median(residuals)
    mad = np.median(np.abs(residuals - center))
    if mad <= 0:
        mad = np.std(residuals)
    if mad <= 0:
        return data.loc[valid].copy()

    keep_valid = np.abs(residuals - center) <= (mad_mult * mad)
    keep_mask = np.zeros(len(data), dtype=bool)
    keep_mask[np.where(valid)[0][keep_valid]] = True
    return data.loc[keep_mask].copy()


def knn_density_estimation(
    data: np.ndarray,
    n_neighbors: int = 3,
    small_num: float = 1e-6
) -> np.ndarray:
    """
    Estimate density using K-Nearest Neighbors algorithm.
    
    Parameters
    ----------
    data : np.ndarray
        Data points as 2D array (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors for KNN (default: 3).
    small_num : float, optional
        Small constant to avoid division by zero (default: 1e-6).
    
    Returns
    -------
    np.ndarray
        Density estimates for each point.
    
    Examples
    --------
    >>> xy = np.column_stack([x_values, y_values])
    >>> density = knn_density_estimation(xy)
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    distances, _ = knn.kneighbors(data)
    density = 1 / np.maximum(distances[:, -1], small_num)
    return density


def proximity_gate_knn(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    percent: float = 0.9,
    n_neighbors: int = 3,
    small_num: float = 1e-6,
    return_plot_data: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray], Tuple[float, float]]]:
    """
    Select data points based on KNN density-proximity gating.
    
    Uses 1D KNN density estimation on each axis separately,
    then intersects the selected points from both axes.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    percent : float, optional
        Fraction of points to keep (default: 0.9).
    n_neighbors : int, optional
        Number of neighbors for KNN (default: 3).
    small_num : float, optional
        Small constant to avoid division by zero (default: 1e-6).
    return_plot_data : bool, optional
        If True, return additional data for plotting (default: False).
    
    Returns
    -------
    Union[pd.DataFrame, Tuple]
        If return_plot_data is False: gated DataFrame.
        If return_plot_data is True: (gated_data, [X, Y, Z], (densest_x, densest_y)).
    
    Examples
    --------
    >>> gated = proximity_gate_knn(data, 'SSC-H', 'SSC-A', percent=0.95)
    """
    xvals = data[x_axis].values
    yvals = data[y_axis].values
    
    # 1D density for X axis
    x_1d = xvals.reshape(-1, 1)
    knn_x = NearestNeighbors(n_neighbors=n_neighbors)
    knn_x.fit(x_1d)
    distances_x, _ = knn_x.kneighbors(x_1d)
    density_x = 1 / np.maximum(distances_x[:, -1], small_num)
    densest_x_idx = np.argmax(density_x)
    densest_x = xvals[densest_x_idx]
    
    # Select percent closest to densest X
    dist_x = np.abs(xvals - densest_x)
    indices_sorted_x = np.argsort(dist_x)
    cut_x = int(len(indices_sorted_x) * percent)
    set_x = set(indices_sorted_x[:cut_x])
    
    # 1D density for Y axis
    y_1d = yvals.reshape(-1, 1)
    knn_y = NearestNeighbors(n_neighbors=n_neighbors)
    knn_y.fit(y_1d)
    distances_y, _ = knn_y.kneighbors(y_1d)
    density_y = 1 / np.maximum(distances_y[:, -1], small_num)
    densest_y_idx = np.argmax(density_y)
    densest_y = yvals[densest_y_idx]
    
    # Select percent closest to densest Y
    dist_y = np.abs(yvals - densest_y)
    indices_sorted_y = np.argsort(dist_y)
    cut_y = int(len(indices_sorted_y) * percent)
    set_y = set(indices_sorted_y[:cut_y])
    
    # Intersection
    overlap_indices = list(set_x.intersection(set_y))
    gated_data = data.iloc[overlap_indices]
    
    if not return_plot_data:
        return gated_data
    
    # Generate 2D density grid for contour plotting
    xy = np.vstack([xvals, yvals]).T
    knn_xy = NearestNeighbors(n_neighbors=n_neighbors)
    knn_xy.fit(xy)
    x_min, x_max = xvals.min(), xvals.max()
    y_min, y_max = yvals.min(), yvals.max()
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    grid_distances, _ = knn_xy.kneighbors(positions)
    grid_density = 1 / np.maximum(grid_distances[:, -1], small_num)
    Z = np.reshape(grid_density, X.shape)
    
    return gated_data, [X, Y, Z], (densest_x, densest_y)


def proximity_gate_knn_euclidean(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    percent: float = 0.9,
    n_neighbors: int = 3,
    small_num: float = 1e-6,
    return_plot_data: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray], Tuple[float, float]]]:
    """
    Select cells by Euclidean distance from the single densest point.
    
    This method finds the single point with highest 2D density, then selects cells 
    based on their Euclidean distance to this densest point.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    percent : float, optional
        Fraction of points to keep (default: 0.9).
    n_neighbors : int, optional
        Number of neighbors for KNN (default: 3).
    small_num : float, optional
        Small constant to avoid division by zero (default: 1e-6).
    return_plot_data : bool, optional
        If True, return additional data for plotting (default: False).
    
    Returns
    -------
    Union[pd.DataFrame, Tuple]
        If return_plot_data is False: gated DataFrame.
        If return_plot_data is True: (gated_data, [X, Y, Z], (densest_x, densest_y)).
    """
    xvals = data[x_axis].values
    yvals = data[y_axis].values
    xy = np.vstack([xvals, yvals]).T
    
    # 2D KNN density estimation
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(xy)
    distances, _ = knn.kneighbors(xy)
    density = 1 / np.maximum(distances[:, -1], small_num)
    
    # Find densest point
    densest_idx = np.argmax(density)
    densest_x = xvals[densest_idx]
    densest_y = yvals[densest_idx]
    
    # Calculate Euclidean distance from densest point
    euclidean_dist = np.sqrt((xvals - densest_x)**2 + (yvals - densest_y)**2)
    
    # Select closest percent
    sorted_indices = np.argsort(euclidean_dist)
    cutoff = int(len(sorted_indices) * percent)
    selected_indices = sorted_indices[:cutoff]
    gated_data = data.iloc[selected_indices]
    
    if not return_plot_data:
        return gated_data
    
    # Generate 2D density grid for contour plotting
    x_min, x_max = xvals.min(), xvals.max()
    y_min, y_max = yvals.min(), yvals.max()
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    grid_distances, _ = knn.kneighbors(positions)
    grid_density = 1 / np.maximum(grid_distances[:, -1], small_num)
    Z = np.reshape(grid_density, X.shape)
    
    return gated_data, [X, Y, Z], (densest_x, densest_y)


def proximity_gate_knn_density(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    percent: float = 0.9,
    n_neighbors: int = 3,
    small_num: float = 1e-6,
    return_plot_data: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray], Tuple[float, float]]]:
    """
    Select cells by ranking density - highest density cells are selected.
    
    This method performs 2D KNN density estimation and selects cells with the 
    highest density values (top percent).
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    percent : float, optional
        Fraction of points to keep (default: 0.9).
    n_neighbors : int, optional
        Number of neighbors for KNN (default: 3).
    small_num : float, optional
        Small constant to avoid division by zero (default: 1e-6).
    return_plot_data : bool, optional
        If True, return additional data for plotting (default: False).
    
    Returns
    -------
    Union[pd.DataFrame, Tuple]
        If return_plot_data is False: gated DataFrame.
        If return_plot_data is True: (gated_data, [X, Y, Z], (densest_x, densest_y)).
    """
    xvals = data[x_axis].values
    yvals = data[y_axis].values
    xy = np.vstack([xvals, yvals]).T
    
    # 2D KNN density estimation
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(xy)
    distances, _ = knn.kneighbors(xy)
    density = 1 / np.maximum(distances[:, -1], small_num)
    
    # Sort by density and select top percent
    sorted_indices = np.argsort(density)[::-1]  # Highest density first
    cutoff = int(len(sorted_indices) * percent)
    selected_indices = sorted_indices[:cutoff]
    gated_data = data.iloc[selected_indices]
    
    # Find densest point for plot data
    densest_idx = np.argmax(density)
    densest_x = xvals[densest_idx]
    densest_y = yvals[densest_idx]
    
    if not return_plot_data:
        return gated_data
    
    # Generate 2D density grid for contour plotting
    x_min, x_max = xvals.min(), xvals.max()
    y_min, y_max = yvals.min(), yvals.max()
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    grid_distances, _ = knn.kneighbors(positions)
    grid_density = 1 / np.maximum(grid_distances[:, -1], small_num)
    Z = np.reshape(grid_density, X.shape)
    
    return gated_data, [X, Y, Z], (densest_x, densest_y)


def cluster_beads(
    beads_data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    x_cutoff_min: float = 0.5e6,
    x_cutoff_max: float = 8e6,
    y_cutoff_max: float = 16e6,
    n_clusters: int = 3,
    n_init: int = 10,
    selected_clusters: Tuple[int, int] = (0, 1)
) -> Tuple[pd.DataFrame, float, np.ndarray]:
    """
    Cluster beads data using KMeans and compute gating cutoff.
    
    Parameters
    ----------
    beads_data : pd.DataFrame
        Beads FCS data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    x_cutoff_min : float, optional
        Minimum X cutoff for initial filtering (default: 0.5e6).
    x_cutoff_max : float, optional
        Maximum X cutoff for initial filtering (default: 8e6).
    y_cutoff_max : float, optional
        Maximum Y cutoff for initial filtering (default: 16e6).
    n_clusters : int, optional
        Number of clusters for KMeans (default: 3).
    n_init : int, optional
        Number of KMeans initializations (default: 10).
    selected_clusters : Tuple[int, int], optional
        Indices of clusters to use for cutoff calculation (default: (0, 1)).
    
    Returns
    -------
    Tuple[pd.DataFrame, float, np.ndarray]
        (clustered_data, x_axis_cutoff, cluster_centers).
    
    Examples
    --------
    >>> clustered, cutoff, centers = cluster_beads(beads, 'FSC-A', 'SSC-A')
    """
    beads_channels = [x_axis, y_axis]
    
    # Initial filtering
    mask = (
        (beads_data[x_axis] > x_cutoff_min) &
        (beads_data[x_axis] < x_cutoff_max) &
        (beads_data[y_axis] < y_cutoff_max)
    )
    beads_filtered = beads_data[mask].copy()
    
    if beads_filtered.empty:
        raise ValueError('No bead events remain after initial filtering')

    # Prefer sklearn KMeans, but fall back to SciPy on Windows environments
    # where threadpoolctl can fail while inspecting native libraries.
    bead_array = beads_filtered[beads_channels].to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    try:
        labels = kmeans.fit_predict(bead_array)
        centers = kmeans.cluster_centers_
    except OSError:
        if len(bead_array) < n_clusters:
            raise ValueError('Not enough bead events available for clustering')

        seed_positions = np.linspace(0, len(bead_array) - 1, n_clusters, dtype=int)
        initial_centers = bead_array[np.argsort(bead_array[:, 0])][seed_positions]
        centers, labels = kmeans2(bead_array, initial_centers, minit='matrix')

    beads_filtered.loc[:, 'cluster'] = labels
    
    # Sort centers by Y axis
    sorted_centers = centers[centers[:, 1].argsort()]
    sorted_centers_df = pd.DataFrame(sorted_centers, columns=beads_channels)
    
    # Calculate cutoff
    lower_idx, upper_idx = selected_clusters
    if lower_idx < 0 or upper_idx >= len(sorted_centers_df) or lower_idx >= upper_idx:
        raise ValueError("Invalid selected_clusters indices")
    
    lower_value = sorted_centers_df[x_axis][lower_idx]
    upper_value = sorted_centers_df[x_axis][upper_idx]
    x_axis_cutoff = (upper_value - lower_value) / 2 + lower_value
    
    return beads_filtered, x_axis_cutoff, centers


def find_local_minimum_kde(
    kde_x: np.ndarray,
    kde_y: np.ndarray,
    min_distance: int = 50
) -> Tuple[float, List[int]]:
    """
    Find local minimum between two highest peaks in KDE data.
    
    Parameters
    ----------
    kde_x : np.ndarray
        X values from KDE plot.
    kde_y : np.ndarray
        Y values (density) from KDE plot.
    min_distance : int, optional
        Minimum index distance between peaks (default: 50).
    
    Returns
    -------
    Tuple[float, List[int]]
        (minimum_x_value, [peak1_index, peak2_index]).
    
    Raises
    ------
    ValueError
        If two peaks cannot be found with given min_distance.
    
    Examples
    --------
    >>> min_val, peaks = find_local_minimum_kde(x_data, y_data)
    """
    peaks, _ = find_peaks(kde_y)
    peak_values = kde_y[peaks]
    
    # Sort peaks by value descending
    sorted_peak_indices = np.argsort(peak_values)[::-1]
    highest_peaks = []
    
    for i in range(len(sorted_peak_indices)):
        if len(highest_peaks) == 0:
            highest_peaks.append(peaks[sorted_peak_indices[i]])
        else:
            if abs(peaks[sorted_peak_indices[i]] - highest_peaks[0]) > min_distance:
                highest_peaks.append(peaks[sorted_peak_indices[i]])
                break
    
    if len(highest_peaks) < 2:
        raise ValueError(
            f"Could not find two peaks with min distance {min_distance} "
            f"within {len(sorted_peak_indices)} peaks"
        )
    
    # Find local minimum between peaks
    min_idx = np.argmin(kde_y[min(highest_peaks):max(highest_peaks)]) + min(highest_peaks)
    min_value = kde_x[min_idx]
    
    return min_value, highest_peaks


def filter_by_density_sd(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    n_neighbors: int = 5,
    sd_multiplier: float = 2.0,
    log_transform: bool = False,
    small_num: float = 1e-6
) -> pd.DataFrame:
    """
    Filter data by removing outliers based on density standard deviation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    n_neighbors : int, optional
        Number of neighbors for KNN density (default: 5).
    sd_multiplier : float, optional
        Number of standard deviations for cutoff (default: 2.0).
    log_transform : bool, optional
        Whether to log-transform density (default: False).
    small_num : float, optional
        Small constant for numerical stability (default: 1e-6).
    
    Returns
    -------
    pd.DataFrame
        Filtered data.
    
    Examples
    --------
    >>> filtered = filter_by_density_sd(data, 'SSC-H', 'SSC-A', sd_multiplier=2)
    """
    xy = np.vstack([data[x_axis], data[y_axis]]).T
    
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(xy)
    distances, _ = knn.kneighbors(xy)
    density = 1 / distances[:, -1]
    
    if log_transform:
        density = np.log(density + small_num)
    
    mean_density = np.mean(density)
    std_density = np.std(density)
    
    lower_bound = mean_density - sd_multiplier * std_density
    upper_bound = mean_density + sd_multiplier * std_density
    
    mask = (density >= lower_bound) & (density <= upper_bound)
    filtered_data = data[mask].copy()
    
    # Additional filter for positive values
    filtered_data = filtered_data[
        (filtered_data[x_axis] > 0) & (filtered_data[y_axis] > 0)
    ]
    
    return filtered_data


def derive_gate_value(
    ref_data: pd.Series,
    method: str = 'max',
    percentile_q: float = 0.99,
    k_sigma: float = 2.5,
) -> float:
    """
    Derive a gate threshold from a reference population.

    Parameters
    ----------
    ref_data : pd.Series
        1-D channel values from the proximity-gated reference.
    method : str
        Gate derivation strategy:
        - 'max'        : maximum value (current default)
        - 'percentile' : ``ref_data.quantile(percentile_q)``
        - 'mean_ksd'   : ``mean + k_sigma * std``
    percentile_q : float
        Quantile used when *method='percentile'* (default 0.99).
    k_sigma : float
        Number of standard deviations when *method='mean_ksd'* (default 2.5).

    Returns
    -------
    float
        Gate value.
    """
    if ref_data.empty:
        return 0.0

    if method == 'percentile':
        return float(ref_data.quantile(percentile_q))
    elif method == 'mean_ksd':
        return float(ref_data.mean() + k_sigma * ref_data.std())
    else:  # 'max'
        return float(ref_data.max())


def calculate_quadrant_percentages(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    x_gate: float,
    y_gate: float
) -> Dict[str, float]:
    """
    Calculate percentage of data in each quadrant.
    
    Quadrants are defined as:
    - Q1: x > x_gate, y > y_gate (upper right)
    - Q2: x <= x_gate, y > y_gate (upper left)
    - Q3: x <= x_gate, y <= y_gate (lower left)
    - Q4: x > x_gate, y <= y_gate (lower right)
    
    Parameters
    ----------
    data : pd.DataFrame
        Flow cytometry data.
    x_axis : str
        Column name for X axis.
    y_axis : str
        Column name for Y axis.
    x_gate : float
        X axis gate value.
    y_gate : float
        Y axis gate value.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with quadrant percentages (0-1).
    
    Examples
    --------
    >>> percentages = calculate_quadrant_percentages(data, 'V450-A', '7-AAD', 1e3, 1e3)
    """
    total = len(data)
    if total == 0:
        return {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}
    
    q1 = len(data[(data[x_axis] > x_gate) & (data[y_axis] > y_gate)]) / total
    q2 = len(data[(data[x_axis] <= x_gate) & (data[y_axis] > y_gate)]) / total
    q3 = len(data[(data[x_axis] <= x_gate) & (data[y_axis] <= y_gate)]) / total
    q4 = len(data[(data[x_axis] > x_gate) & (data[y_axis] <= y_gate)]) / total
    
    return {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}


def calculate_cytotoxicity(
    target_live_percent: float,
    target_baseline_live: float,
    to_zero: bool = True
) -> float:
    """
    Calculate cytotoxicity percentage.
    
    Parameters
    ----------
    target_live_percent : float
        Percentage of live target cells in coculture.
    target_baseline_live : float
        Baseline percentage of live target cells.
    to_zero : bool, optional
        If True, negative values are set to 0 (default: True).
    
    Returns
    -------
    float
        Cytotoxicity percentage.
    
    Examples
    --------
    >>> cytotox = calculate_cytotoxicity(0.75, 0.95)
    """
    if target_baseline_live <= 0:
        return 0.0
    
    # Convert baseline to percentage
    baseline_percent = target_baseline_live * 100
    
    # Calculate dead cells in sample and baseline
    dead_in_sample = 100 - target_live_percent
    dead_in_baseline = 100 - baseline_percent
    
    # Calculate cytotoxicity as percentage of excess death
    cytotox = ((dead_in_sample - dead_in_baseline) / baseline_percent) * 100
    
    if to_zero and cytotox < 0:
        cytotox = 0.0
    
    return cytotox


def concordance_correlation_coefficient(
    y1: np.ndarray,
    y2: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Lin's Concordance Correlation Coefficient (CCC).

    Measures agreement between two continuous measurements, combining
    both precision (Pearson r) and accuracy (bias correction factor).

    Parameters
    ----------
    y1, y2 : array-like
        Paired measurements (same length, no NaN).

    Returns
    -------
    ccc : float
        Concordance correlation coefficient (-1 to 1).
    precision : float
        Pearson r component.
    accuracy : float
        Bias correction factor C_b.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if len(y1) < 3:
        return np.nan, np.nan, np.nan

    mu1, mu2 = y1.mean(), y2.mean()
    s1, s2 = y1.std(ddof=1), y2.std(ddof=1)
    if s1 == 0 or s2 == 0:
        return np.nan, np.nan, np.nan

    r = np.corrcoef(y1, y2)[0, 1]  # Pearson r (precision)
    # Bias correction factor
    c_b = (2 * s1 * s2) / (s1**2 + s2**2 + (mu1 - mu2)**2)
    ccc = r * c_b
    return float(ccc), float(r), float(c_b)
