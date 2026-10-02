# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Explicit recurrence selection and user-defined adapter loading."""
from importlib import import_module
from .base import CalibrationAdapter
from .mamba2 import Mamba2Adapter
from .gdn import GDNAdapter
from .kda import KDAAdapter

_BUILTINS = {'mamba2': Mamba2Adapter, 'gdn': GDNAdapter, 'kda': KDAAdapter}


def load_adapter(name='auto', *, family=None):
    """Load a built-in name or an explicit importable ``module:Class``.

    Auto uses declared family metadata only. It does not guess from a model ID
    or download/import a serving engine. Custom modules are user-supplied code.
    """
    if name in (None, 'auto'):
        if family not in _BUILTINS:
            raise ValueError('Auto requires a declared mamba2/gdn/kda family; otherwise specify an adapter')
        name = family
    if name in _BUILTINS:
        cls = _BUILTINS[name]
    elif isinstance(name, str) and name.count(':') == 1:
        module, attribute = name.split(':')
        if not module or not attribute:
            raise ValueError('Use module:AdapterClass for a custom adapter')
        cls = getattr(import_module(module), attribute)
    else:
        raise ValueError(f'Unknown adapter {name!r}; use mamba2, gdn, kda or module:AdapterClass')
    if not isinstance(cls, type) or not issubclass(cls, CalibrationAdapter):
        raise TypeError('Custom adapters must subclass CalibrationAdapter')
    adapter = cls()
    if not isinstance(getattr(adapter, 'family', None), str) or not adapter.family:
        raise TypeError('Adapter must declare a nonempty family')
    if type(getattr(adapter, 'erase', None)) is not bool:
        raise TypeError('Adapter must declare boolean erase')
    if family is not None and family != adapter.family:
        raise ValueError(f'Adapter family {adapter.family!r} differs from source family {family!r}')
    return adapter
