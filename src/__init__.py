"""AutoCytotox implementation package."""

__version__ = "1.0.0"
__author__ = "Aleksander Szarzynski"

# Python 3.10+ compatibility fix for FlowCytometryTools
import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

# Lazy exports — avoid importing heavy modules (scipy, seaborn, etc.)
# at package init time. This prevents crashes when multiprocessing 'spawn'
# workers do `import src.paths` and inadvertently trigger the full import
# chain. Users can still do `from src import CytotoxEvaluator` etc.; the
# actual import is deferred to first access.

__all__ = [
    'CytotoxEvaluator',
    'ChannelConfig',
    'GatingConfig',
    'PlotConfig',
    'PlotSettings',
    'get_settings',
    'set_settings',
]


def __getattr__(name):
    if name in ('CytotoxEvaluator', 'ChannelConfig', 'GatingConfig', 'PlotConfig'):
        from .cytotox_evaluator import CytotoxEvaluator, ChannelConfig, GatingConfig, PlotConfig
        return locals()[name]
    if name in ('PlotSettings', 'get_settings', 'set_settings'):
        from .plotting import PlotSettings, get_settings, set_settings
        return locals()[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
