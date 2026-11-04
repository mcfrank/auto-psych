"""Saved out-of-fold scores and pool predictions (src/rsa_open/loop/scores.py; the review's A1)."""

import numpy as np

from src.rsa.fit import FitSettings


def _store(tmp_path, folds=None, settings=FitSettings(dense_mass=True)):
    from src.rsa_open.loop.scores import ScoreStore

    corpus = tmp_path / "corpus.json"
    if not corpus.exists():
        corpus.write_text('{"blocks": []}')
    folds = folds if folds is not None else {"E": np.array([0, 1, 0, 1]), "exp1": np.array([1, 0])}
    return ScoreStore(tmp_path / ".scores", corpus, folds, settings, FitSettings(num_warmup=300, num_samples=500),
                      1597, 100)


def test_scores_come_back_exactly_and_a_failed_fold_with_its_reason(tmp_path):
    from src.rsa_open.corpus import CorpusCV

    store, theory = _store(tmp_path), tmp_path / "t.py"
    theory.write_text("# a theory\n")
    assert store.load_cv("t", theory, 600.0) is None
    cv = CorpusCV({"E": np.array([-0.1, -0.2, -0.3, -0.4]), "exp1": np.array([-1.5, -2.5])}, False)
    store.save_cv("t", theory, 600.0, cv, "cross-validation folds did not converge")
    back, problem = store.load_cv("t", theory, 600.0)
    assert problem == "cross-validation folds did not converge" and back.converged is False
    assert all(np.array_equal(back.pointwise[b], cv.pointwise[b]) for b in cv.pointwise)
    store.save_cv("u", theory, 600.0, None, "a cross-validation fold failed: ValueError")
    assert store.load_cv("u", theory, 600.0) == (None, "a cross-validation fold failed: ValueError")
    preds = np.linspace(0, 1, 17)
    store.save_preds("t", theory, preds)
    assert np.array_equal(store.load_preds("t", theory), preds)


def test_a_result_is_never_read_for_anything_it_does_not_match(tmp_path):
    from src.rsa_open.corpus import CorpusCV

    store, theory = _store(tmp_path), tmp_path / "t.py"
    theory.write_text("# a theory\n")
    store.save_cv("t", theory, 600.0, CorpusCV({"E": np.zeros(4)}, True), None)
    store.save_preds("t", theory, np.ones(3))
    assert store.load_cv("t", theory, 900.0) is None  # another fold time limit
    other = _store(tmp_path, folds={"E": np.array([1, 0, 1, 0]), "exp1": np.array([1, 0])})
    assert other.load_cv("t", theory, 600.0) is None  # another fold assignment
    assert _store(tmp_path, settings=FitSettings(seed=1)).load_preds("t", theory) is None  # another fit
    theory.write_text("# a theory, changed\n")
    assert store.load_cv("t", theory, 600.0) is None and store.load_preds("t", theory) is None


def test_a_result_without_its_record_is_not_complete(tmp_path):
    store, theory = _store(tmp_path), tmp_path / "t.py"
    theory.write_text("# a theory\n")
    store.save_preds("t", theory, np.ones(3))
    next((tmp_path / ".scores").glob("t.*.json")).unlink()  # killed between the two writes
    assert store.load_preds("t", theory) is None
