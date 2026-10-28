"""What the ablation of chain 0's final needs to know about it (`src.rsa.ablation.variants`).

The switchable model itself is `final_c0.py`, a memo model file, read as text and never imported
(the loader injects its OBJ/UTT domains per display shape). The names here follow the
sr-oct26 ablation's switchable finals (`src.ablation.variants`).

| mechanism | parameter | removed |
|---|---|---|
| depth2 | | the speaker reasons about L0, not L1 |
| interior | w_interior | no interior suppression |
| avoidable | c_avoid | no avoidable-ambiguity cost |
| confusion | w_confusion | no confusion penalty |
| singleton, distinct, features, typicality | w_singleton, w_distinct, w_features, w_typical | that salience term leaves the prior |
| valence | w_valence | the feature weight no longer shifts with the framing's valence |
| color | w_color | no colour salience (constant on the campaign's pooled data: no greyscale display) |
| attention | logit_p_base | the listener always attends |
| popout, load, candidate | w_popout, w_load, w_candidate | that term leaves the attention gate (needs attention) |
| inhibit_leak | w_inhibit_leak | inattention is captured by matching objects only (needs attention) |
| outlier | logit_p_outlier | no solitary-foil outlier |

``popout`` is inert in the final (``INERT``): it raises attention only when the word fits one
object, and there the attending listener and the inattentive one (captured by the matching
objects) choose that same object, so no prediction moves (by more than ~3e-6). The final carries
its parameter to no effect; the elimination can drop it at no cost.

``alpha`` and ``lapse`` are the core. The lapse is not removable here, unlike the sr-oct26
finals': without it (and without attention's capture term) an object a word is false of has
probability exactly 0, and a fit refuses a model that gives an observed choice zero probability.
"""

from pathlib import Path

FINAL = "boundary_avoidable_ambiguity_listener"
SOURCE = Path(__file__).with_name("final_c0.py")

MECHANISMS = ("depth2", "interior", "avoidable", "confusion", "singleton", "distinct", "features", "valence",
              "color", "typicality", "attention", "popout", "load", "candidate", "inhibit_leak", "outlier")
REQUIRES = {m: ("attention",) for m in ("popout", "load", "candidate", "inhibit_leak")}
KEEP_ONE_OF = ()
INERT = ("popout",)
CORE = ("alpha", "lapse")
PARAMETERS = {"depth2": (), "interior": ("w_interior",), "avoidable": ("c_avoid",), "confusion": ("w_confusion",),
              "singleton": ("w_singleton",), "distinct": ("w_distinct",), "features": ("w_features",),
              "valence": ("w_valence",), "color": ("w_color",), "typicality": ("w_typical",),
              "attention": ("logit_p_base",), "popout": ("w_popout",), "load": ("w_load",),
              "candidate": ("w_candidate",), "inhibit_leak": ("w_inhibit_leak",), "outlier": ("logit_p_outlier",)}
SHORT = {"depth2": "d2", "interior": "int", "avoidable": "avo", "confusion": "conf", "singleton": "sing",
         "distinct": "dist", "features": "feat", "valence": "val", "color": "col", "typicality": "typ",
         "attention": "att", "popout": "pop", "load": "load", "candidate": "cand", "inhibit_leak": "leak",
         "outlier": "out"}
# Each mechanism's parameter at the value that turns it off (the tests check that it does):
# weights at 0; the gates at a logit whose sigmoid is 1 (attend) or 0 (no outlier) in float32;
# the leak at a softplus argument whose value is ~1e-44.
NEUTRAL = {"w_interior": 0.0, "c_avoid": 0.0, "w_confusion": 0.0, "w_singleton": 0.0, "w_distinct": 0.0,
           "w_features": 0.0, "w_valence": 0.0, "w_color": 0.0, "w_typical": 0.0, "logit_p_base": 100.0,
           "w_popout": 0.0, "w_load": 0.0, "w_candidate": 0.0, "w_inhibit_leak": -100.0, "logit_p_outlier": -100.0}
