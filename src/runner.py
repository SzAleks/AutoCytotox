"""Compatibility wrapper for the workflow runner implementation."""

from .workflow.runner import *
from .workflow.runner import run_from_config

__all__ = ['run_from_config']
