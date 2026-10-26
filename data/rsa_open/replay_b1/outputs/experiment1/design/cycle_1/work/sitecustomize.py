import sys
import types
from types import SimpleNamespace

if "src.rsa.experiment.design" not in sys.modules:
    def _catch_spec(shape, rng):
        n_obj, n_feat = shape
        word = int(rng.integers(n_feat))
        target = int(rng.integers(n_obj))
        objects = []
        for i in range(n_obj):
            row = [0] * n_feat
            if i == target:
                row[word] = 1
            else:
                other_f = (word + 1 + (i % max(1, n_feat - 1))) % n_feat
                if other_f != word:
                    row[other_f] = 1
            objects.append(tuple(row))
        return SimpleNamespace(objects=tuple(objects), utterance=word), target

    exp_mod = types.ModuleType("src.rsa.experiment")
    des_mod = types.ModuleType("src.rsa.experiment.design")
    des_mod._catch_spec = _catch_spec
    exp_mod.design = des_mod
    sys.modules["src.rsa.experiment"] = exp_mod
    sys.modules["src.rsa.experiment.design"] = des_mod
