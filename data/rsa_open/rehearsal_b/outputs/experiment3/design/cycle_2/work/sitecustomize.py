import sys
import types

try:
    import src.rsa.experiment.design  # noqa: F401
except ModuleNotFoundError:
    from src.rsa.context import Context
    from src.rsa.design_space import feature_names, games

    def _catch_spec(shape, rng):
        n_obj, n_feat = int(shape[0]), int(shape[1])
        candidates = []
        fnames = feature_names(n_feat)
        pool = games(n_obj, n_feat, synonyms=False) or games(n_obj, n_feat, synonyms=True)
        for matrix in pool:
            for u in range(n_feat):
                holders = [r for r, row in enumerate(matrix) if row[u] == 1]
                if len(holders) == 1:
                    candidates.append((Context(objects=matrix, feature_names=fnames, utterance=u), holders[0]))
        idx = int(rng.integers(len(candidates)))
        return candidates[idx]

    exp_mod = types.ModuleType("src.rsa.experiment")
    des_mod = types.ModuleType("src.rsa.experiment.design")
    des_mod._catch_spec = _catch_spec
    exp_mod.design = des_mod
    sys.modules["src.rsa.experiment"] = exp_mod
    sys.modules["src.rsa.experiment.design"] = des_mod
