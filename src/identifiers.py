"""
Filename Identifier Strategies for the autocytotox Autogating Pipeline.

This module provides built-in strategies for mapping FCS filenames to group
labels (e.g. by well row or full well position) and a loader that resolves
user-supplied callable strategies referenced as ``module.py:function``.

Author: Aleksander Szarzynski TUW 2026
"""

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Callable, Dict, List, TypeAlias


IdentifierHandler: TypeAlias = Callable[[List[str]], List[str]]


def _last_dash_token(filename: str) -> str:
    stem = Path(filename).stem
    token = stem.split('-')[-1]
    if not token:
        raise ValueError(f'Could not derive identifier token from filename: {filename}')
    return token


def identifier_by_well_row(filename_list: List[str]) -> List[str]:
    return [_last_dash_token(filename)[0] for filename in filename_list]


def identifier_by_well_position(filename_list: List[str]) -> List[str]:
    return [_last_dash_token(filename) for filename in filename_list]


def identifier_by_stem(filename_list: List[str]) -> List[str]:
    return [Path(filename).stem for filename in filename_list]


BUILTIN_IDENTIFIER_STRATEGIES: Dict[str, IdentifierHandler] = {
    'well_row': identifier_by_well_row,
    'well_position': identifier_by_well_position,
    'stem': identifier_by_stem,
}


def get_builtin_identifier_strategy_names() -> List[str]:
    return sorted(BUILTIN_IDENTIFIER_STRATEGIES)


def preview_identifier_groups(
    filename_list: List[str],
    identifier_handler: IdentifierHandler,
) -> Dict[str, List[str]]:
    identifier_list = identifier_handler(filename_list)
    preview: Dict[str, List[str]] = {}
    for identifier, filename in zip(identifier_list, filename_list):
        preview.setdefault(identifier, []).append(filename)
    return preview


def _load_callable_from_path(callable_path: str, repo_root: str | None = None) -> IdentifierHandler:
    module_spec, _, attribute_name = callable_path.partition(':')
    if not module_spec or not attribute_name:
        raise ValueError(
            'Identifier callable must use the format "module:function" or "path/to/file.py:function".'
        )

    if module_spec.endswith('.py') or os.path.sep in module_spec or '/' in module_spec:
        module_path = module_spec
        if not os.path.isabs(module_path):
            module_path = os.path.join(repo_root or os.getcwd(), module_path)
        if not os.path.exists(module_path):
            raise FileNotFoundError(f'Identifier callable module not found: {module_path}')

        module_name = f'_autocytotox_identifier_{abs(hash(os.path.abspath(module_path)))}'
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not import identifier callable module: {module_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    handler = getattr(module, attribute_name, None)
    if handler is None or not callable(handler):
        raise ValueError(f'Identifier callable {callable_path!r} does not resolve to a callable object.')
    return handler


def resolve_identifier_handler(
    strategy: str = 'well_row',
    callable_path: str | None = None,
    repo_root: str | None = None,
) -> IdentifierHandler:
    if callable_path:
        return _load_callable_from_path(callable_path, repo_root=repo_root)

    handler = BUILTIN_IDENTIFIER_STRATEGIES.get(strategy)
    if handler is None:
        available = ', '.join(get_builtin_identifier_strategy_names())
        raise ValueError(f'Unknown identifier strategy {strategy!r}. Available: {available}')
    return handler