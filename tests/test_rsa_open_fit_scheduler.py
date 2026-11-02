"""The fit children's memory scheduler (src/rsa_open/loop/fitting.py, FitScheduler)."""

import threading
import time

GB = 2**30


def _holder(sched, name, entered, release, at_least=0.0):
    def run():
        with sched.room(name, at_least):
            entered.set()
            release.wait(10)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_a_fit_waits_until_the_memory_reserved_for_the_running_ones_leaves_room_for_it():
    from src.rsa_open.loop.fitting import FitScheduler

    sched = FitScheduler(25 * GB, slots=8, rss=lambda: 0)
    sched.observe("heavy", 8 * GB)  # reserved at 1.25 x its peak: 10 GB a fit
    release = threading.Event()
    first, second, third = threading.Event(), threading.Event(), threading.Event()
    _holder(sched, "heavy", first, release)
    _holder(sched, "heavy", second, release)
    assert first.wait(5) and second.wait(5)
    t = _holder(sched, "heavy", third, threading.Event())
    assert not third.wait(0.5)  # 30 GB would not fit in 25
    release.set()
    assert third.wait(5)
    t.join(0)


def test_this_process_s_own_memory_counts_against_the_room():
    from src.rsa_open.loop.fitting import FitScheduler

    sched = FitScheduler(25 * GB, slots=8, rss=lambda: 23 * GB)
    sched.observe("light", 1 * GB)
    release, first, second = threading.Event(), threading.Event(), threading.Event()
    _holder(sched, "light", first, release)
    assert first.wait(5)  # one fit always runs
    _holder(sched, "light", second, threading.Event())
    assert not second.wait(0.5)  # the loop's own 23 GB + 2 x 1.25 GB reserved > 25 GB
    release.set()
    assert second.wait(5)


def test_a_theory_needing_more_than_the_job_runs_alone_and_the_slots_bound_the_count():
    from src.rsa_open.loop.fitting import FitScheduler

    sched = FitScheduler(10 * GB, slots=2, rss=lambda: 0)
    huge = threading.Event()
    release = threading.Event()
    _holder(sched, "huge", huge, release, at_least=50 * GB)
    assert huge.wait(5) and sched.running() == 1
    small = threading.Event()
    _holder(sched, "small", small, threading.Event())
    assert not small.wait(0.5)  # nothing joins a fit that has the whole job
    release.set()
    assert small.wait(5)

    sched = FitScheduler(1000 * GB, slots=2, rss=lambda: 0)
    hold = threading.Event()
    events = [threading.Event() for _ in range(3)]
    for e in events:
        _holder(sched, "x", e, hold)
    time.sleep(0.5)
    assert sum(e.is_set() for e in events) == 2  # two slots
    hold.set()


def test_an_unknown_theory_is_given_the_default_room_and_a_known_one_its_peak_with_a_margin():
    from src.rsa_open.loop.fitting import DEFAULT_FIT_BYTES, PEAK_MARGIN, FitScheduler

    sched = FitScheduler(100 * GB, slots=4, rss=lambda: 0)
    assert sched.estimate("new") == DEFAULT_FIT_BYTES
    sched.observe("seen", 4 * GB)
    sched.observe("seen", 3 * GB)  # the largest peak is kept
    assert sched.estimate("seen") == PEAK_MARGIN * 4 * GB


def test_the_job_s_memory_is_slurm_s_allocation(monkeypatch):
    from src.rsa_open.loop import fitting

    monkeypatch.setenv("SLURM_MEM_PER_NODE", "184320")
    assert fitting.job_memory_bytes() == 184320 * 2**20
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "8000")
    monkeypatch.setattr(fitting, "job_cpus", lambda: list(range(24)))
    assert fitting.job_memory_bytes() == 8000 * 2**20 * 24
