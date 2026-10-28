"""Make PyTensor's ``config.change_flags`` apply to the calling thread only.

``change_flags`` saves a config value, sets a new one and restores the saved one
on exit, all on the process-global config. ``infer_static_shape`` (run whenever
a random variable or ``Alloc`` is built) wraps itself in ``change_flags(cxx="")``,
so threads building model graphs at once interleave those saves and restores:
one can save another's temporary ``""`` and restore it last, leaving ``cxx``
empty for good, after which every function the process compiles runs as Python
thunks. The harness builds graphs in threads (a wave's prefits, the background
fold fits, admission), and its runs stalled for hours that way.

With ``make_change_flags_thread_local`` a ``change_flags`` override is held per
thread: the thread that set it sees it, every other thread sees the global
value, exactly what each would see running alone.
"""

from __future__ import annotations

import threading
from typing import Any

_local = threading.local()


def _overrides() -> dict:
    if not hasattr(_local, "overrides"):
        _local.overrides = {}  # config name -> this thread's stack of values
    return _local.overrides


def make_change_flags_thread_local() -> None:
    from pytensor import configparser

    param, flags = configparser.ConfigParam, configparser._ChangeFlagsDecorator
    if getattr(flags, "_thread_local", False):
        return
    global_get = param.__get__

    def get(self: Any, cls: Any, type_: Any, delete_key: bool = False) -> Any:
        stack = _overrides().get(self.name) if cls is not None else None
        if stack:
            return stack[-1]
        return global_get(self, cls, type_, delete_key)

    def enter(self: Any) -> None:
        values = {}
        for k, conf in self.confs.items():
            if not conf.mutable and hasattr(conf, "val"):
                raise Exception(
                    f"Can't change the value of {conf.name} config parameter after initialization!"
                )
            values[k] = conf.apply(self.new_vals[k])
            conf.validate(values[k])
        for k, value in values.items():
            _overrides().setdefault(self.confs[k].name, []).append(value)

    def exit(self: Any, *args: Any) -> None:
        for conf in self.confs.values():
            _overrides()[conf.name].pop()

    param.__get__ = get
    flags.__enter__, flags.__exit__ = enter, exit
    flags._thread_local = True
