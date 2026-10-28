"""PyTensor's ``config.change_flags`` saves and restores a process-global value,
and ``infer_static_shape`` (run whenever a random variable is built) wraps itself
in ``change_flags(cxx="")``. Threads building model graphs at once could
interleave those saves and restores and leave ``cxx`` empty for good, after which
every function the process compiled ran as Python thunks, thousands of times
slower. The harness builds graphs in threads, so it must see per-thread flags."""

import sys
import threading

import pytensor
import pytensor.tensor as pt

from src.runtime.pytensor_threads import make_change_flags_thread_local

make_change_flags_thread_local()  # what run_pymc_inner_loop does before it starts threads


def _build_random_graphs(stop: threading.Event) -> None:
    n = pt.iscalar("n")
    while not stop.is_set():
        pt.random.normal(0, 1, size=(n + 1,))  # symbolic size -> infer_static_shape


def test_concurrent_graph_building_never_strips_the_c_compiler():
    cxx = pytensor.config.cxx
    assert cxx, "this test needs a C compiler"
    x = pt.dvector("x")
    stop = threading.Event()
    builders = [threading.Thread(target=_build_random_graphs, args=(stop,)) for _ in range(3)]
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads inside every change_flags block
    try:
        for t in builders:
            t.start()
        python_thunks = 0
        for _ in range(20):
            fn = pytensor.function([x], pt.exp(x) * 2 + 1)
            python_thunks += sum(not hasattr(t, "cthunk") for t in fn.vm.thunks)
    finally:
        stop.set()
        for t in builders:
            t.join()
        sys.setswitchinterval(interval)
    assert python_thunks == 0
    assert pytensor.config.cxx == cxx


def test_change_flags_applies_within_its_own_thread_only():
    cxx = pytensor.config.cxx
    assert cxx, "this test needs a C compiler"
    inside, seen_elsewhere = [], []
    entered, checked = threading.Event(), threading.Event()

    def flagged():
        with pytensor.config.change_flags(cxx=""):
            inside.append(pytensor.config.cxx)
            entered.set()
            checked.wait()

    t = threading.Thread(target=flagged)
    t.start()
    entered.wait()
    seen_elsewhere.append(pytensor.config.cxx)
    checked.set()
    t.join()
    assert inside == [""]
    assert seen_elsewhere == [cxx]
    assert pytensor.config.cxx == cxx
