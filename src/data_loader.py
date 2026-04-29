"""
Data Loader Module for Flow Cytometry Analysis.

This module provides utilities for loading and organizing FCS (Flow Cytometry Standard)
files from directories. It handles data loading, identifier mapping, and basic
data structure organization.

Author: Aleksander Szarzynski TUW 2026
"""

import os
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

# Python 3.10+ compatibility fix for FlowCytometryTools
import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

from FlowCytometryTools import FCMeasurement


def load_fcs_file(file_path: str, file_id: str = None) -> pd.DataFrame:
    """
    Load a single FCS file and return its data as a DataFrame.
    
    Parameters
    ----------
    file_path : str
        Full path to the FCS file.
    file_id : str, optional
        Identifier for the measurement. If None, filename is used.
    
    Returns
    -------
    pd.DataFrame
        Flow cytometry data with channel columns.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read as FCS format.
    
    Examples
    --------
    >>> data = load_fcs_file('/path/to/sample.fcs')
    >>> print(data.head())
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FCS file not found: {file_path}")
    
    if file_id is None:
        file_id = os.path.basename(file_path)
    
    fcs_data = FCMeasurement(ID=file_id, datafile=file_path)
    return fcs_data


def get_fcs_files_from_folder(folder_path: str, exclude_pattern: str = "Beads") -> List[str]:
    """
    Get list of FCS files from a folder, optionally excluding files matching a pattern.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing FCS files.
    exclude_pattern : str, optional
        Pattern to exclude from file list (default: "Beads").
    
    Returns
    -------
    List[str]
        List of filenames (not full paths) of FCS files.
    
    Examples
    --------
    >>> files = get_fcs_files_from_folder('/path/to/data')
    >>> print(files)
    ['sample1.fcs', 'sample2.fcs']
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    content = os.listdir(folder_path)
    fcs_files = [f for f in content if f.endswith('.fcs') and exclude_pattern not in f]
    return fcs_files


def get_beads_file(folder_path: str, beads_pattern: str = "Beads") -> Optional[str]:
    """
    Find the beads calibration file in a folder.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing FCS files.
    beads_pattern : str, optional
        Pattern to match beads file (default: "Beads").
    
    Returns
    -------
    Optional[str]
        Filename of beads file if found, None otherwise.
    
    Raises
    ------
    ValueError
        If multiple beads files are found.
    
    Examples
    --------
    >>> beads = get_beads_file('/path/to/data')
    >>> print(beads)
    'Beads.fcs'
    """
    content = os.listdir(folder_path)
    beads_files = [f for f in content if beads_pattern in f and f.endswith('.fcs')]
    
    if len(beads_files) == 0:
        return None
    elif len(beads_files) > 1:
        raise ValueError(f"Multiple beads files found in {folder_path}: {beads_files}")
    
    return beads_files[0]


def build_identifier_map(
    filename_list: List[str],
    identifier_handler: Union[Callable[[List[str]], List[str]], List[str]]
) -> Dict[str, List[str]]:
    """
    Build a mapping from identifiers to filenames.
    
    Parameters
    ----------
    filename_list : List[str]
        List of FCS filenames.
    identifier_handler : Union[Callable, List[str]]
        Either a function that extracts identifiers from filenames,
        or a list of identifiers matching the filename_list order.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping identifiers to lists of filenames.
    
    Raises
    ------
    ValueError
        If identifier_handler produces mismatched length.
    
    Examples
    --------
    >>> files = ['day0_SF1_1.fcs', 'day0_SF1_2.fcs', 'day0_SF2_1.fcs']
    >>> def get_id(files): return [f.split('_')[1] for f in files]
    >>> mapping = build_identifier_map(files, get_id)
    >>> print(mapping)
    {'SF1': ['day0_SF1_1.fcs', 'day0_SF1_2.fcs'], 'SF2': ['day0_SF2_1.fcs']}
    """
    if callable(identifier_handler):
        identifier_list = identifier_handler(filename_list)
    elif isinstance(identifier_handler, list):
        identifier_list = identifier_handler
    else:
        raise ValueError("identifier_handler must be callable or list")
    
    if len(identifier_list) != len(filename_list):
        raise ValueError(
            f"Length mismatch: identifiers ({len(identifier_list)}) "
            f"vs filenames ({len(filename_list)})"
        )
    
    identifier_map = {}
    for identifier, filename in zip(identifier_list, filename_list):
        if identifier not in identifier_map:
            identifier_map[identifier] = []
        identifier_map[identifier].append(filename)
    
    return identifier_map


def load_folder_data(
    folder_path: str,
    identifier_handler: Union[Callable[[List[str]], List[str]], List[str]],
    exclude_pattern: str = "Beads"
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all FCS data from a folder, organized by identifier.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing FCS files.
    identifier_handler : Union[Callable, List[str]]
        Function or list for extracting/providing identifiers.
    exclude_pattern : str, optional
        Pattern to exclude from loading (default: "Beads").
    
    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary: {identifier: {filename: DataFrame}}.
    
    Examples
    --------
    >>> def get_id(files): return [f.split('_')[1] for f in files]
    >>> data = load_folder_data('/path/to/data', get_id)
    >>> print(data.keys())
    dict_keys(['SF1', 'SF2'])
    """
    fcs_files = get_fcs_files_from_folder(folder_path, exclude_pattern)
    identifier_map = build_identifier_map(fcs_files, identifier_handler)
    
    data_dict = {}
    for identifier, files in identifier_map.items():
        data_dict[identifier] = {}
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            data_dict[identifier][filename] = load_fcs_file(file_path, filename)
    
    return data_dict


def get_subfolders(parent_path: str) -> List[str]:
    """
    Get list of subfolders in a directory.
    
    Parameters
    ----------
    parent_path : str
        Path to the parent directory.
    
    Returns
    -------
    List[str]
        List of subfolder names.
    
    Examples
    --------
    >>> subfolders = get_subfolders('/path/to/data')
    >>> print(subfolders)
    ['day0', 'day1', 'day2']
    """
    return [f for f in os.listdir(parent_path) 
            if os.path.isdir(os.path.join(parent_path, f))]


def create_folder(folder_path: str) -> None:
    """
    Create a folder if it doesn't exist.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder to create.
    
    Examples
    --------
    >>> create_folder('/path/to/new_folder')
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
