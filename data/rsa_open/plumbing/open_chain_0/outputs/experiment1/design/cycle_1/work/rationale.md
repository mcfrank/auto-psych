# Experiment 1 Rationale: Graded Referential Belief Allocation (`cand_bet`)

## 1. Scientific Goal and Why This Experiment Matters Most for Generalization

In this study of pragmatic reference, our goal is to arrive at theories of how people interpret and choose referring expressions that **generalize across paradigms** rather than overfitting to a single response format (Yarkoni, 2020). All candidate theories in the current pool were originally developed and evaluated on discrete forced-choice data from the closed campaign (`choice`). However, forced choice compresses the listener's entire posterior belief distribution $P(r \mid u, C)$ into a single winner-take-all click.

This compression creates a severe identifiability bottleneck among the current theories:
1. **Discrete Heuristics vs. Graded Recursive Reasoning**: Non-Bayesian heuristics (`oddity_heuristic_listener`, `lexical_preemption_heuristic`) and recursive RSA models (`rsa_l1`, `rsa_l2`, `isolated_graded_costly_l3`, `softmax_surprisal_isolated_costly_l4`) often agree on the modal (most likely) target in standard reference games, while making sharply divergent predictions about the **relative probability mass** assigned to secondary candidates, competitors, and duplicate objects.
2. **Competitor Confusion vs. Perceptual Isolation vs. Singleton Salience**:
   - `competitor_confusion_speaker` predicts that speakers avoid words shared with visually similar distractors, so listeners hearing a shared word strongly favor referents that are visually distinct from their feature-sharing competitors, and on prior trials favor isolated objects via exponential similarity decay.
   - `oddity_heuristic_listener` predicts a discrete step-function preference for unique singletons (`copy_count <= 1`) over identical duplicates, both when hearing an ambiguous word and on uninformative prior queries, while remaining completely flat across distinct non-duplicate objects regardless of their feature counts or Hamming distances.
   - `rsa_l2_singleton_feat_color_valence_l0` (and its baseline counterparts `rsa_l2_singleton_distinct_feat_color_l0`, `rsa_l2_singleton_feat_color_familiar_l0`) combine discrete singleton status (`is_singleton`), continuous total Hamming distinctiveness (`distinct`), and visual feature complexity (`feature_count`) inside a shared object prior at depth 2, while keeping $L_0$ strictly literal.
   - `isolated_graded_costly_l3` and `softmax_surprisal_isolated_costly_l4` combine minimum Hamming distance isolation (`object_isolation`), graded truth applicability ($\exp(-\gamma(\text{fc}-1))$), and utterance extension costs at recursion depths 3 and 4.

To break this forced-choice degeneracy and test whether theories capture people's actual graded pragmatic inferences, **Experiment 1 introduces a 10-point probability allocation (betting) paradigm (`bet` via the `allocation` Dirichlet-multinomial link)** across multi-object scenes (`[2,2], [3,2], [3,3], [4,2], [4,3], [4,4]`) with both **synonymous/redundant features (`synonyms: true`)** and **uninformative prior queries (`prior_queries: true`)**.

---

## 2. Why This Task, Scene Space, and Response Type

### Response Type: 10-Point Allocation (`bet` / `allocation` link)
Rather than clicking a single referent, participants distribute 10 points across the objects in the scene to reflect how likely each object is to be the speaker's intended target (Frank & Goodman, 2012).
- **Direct observation of the full posterior**: On every single trial, the response vector $(y_1, \dots, y_N)$ with $\sum_i y_i = 10$ probes the entire distribution over all $N$ objects simultaneously—capturing zero-point exclusions (literal falsity), graded preference between the primary pragmatic target and secondary literal competitors, and exact splits across identical twins vs. distinct singletons.
- **High Expected Information Gain**: In our design exploration (`eig --quick`), the 10-point betting paradigm achieved the highest information efficiency across all tested paradigms (**3.792 bits out of 3.907 max bits**, **0.01433 bits/$**, more than **2.2x the required floor** of 0.00650 bits/$), and passed parameter recovery (`link-check`) with `converged=True, recovered=True` across all 3 replicates.

### Scene Space: 2 to 4 Objects, 2 to 4 Features, `prior_queries: true`, `synonyms: true`
- **Multi-object competition (`[3,2]`, `[3,3]`, `[4,2]`, `[4,3]`, `[4,4]`)**: With 3 and 4 objects, a heard word can be true of 2 or 3 objects simultaneously that differ in their overlap with other objects, their total feature count, and whether they have an identical twin in the display.
- **Uninformative Prior Queries (`prior_queries: true`)**: Directly measures the listener's baseline referent prior $P(r \mid C)$ when no word is heard yet:
  - `literal_listener`, `rsa_l1`, `rsa_l2`, and `lexical_preemption_heuristic` predict a strictly **uniform** allocation (e.g., `3, 3, 4` or `2, 3, 3, 2` around $1/N$).
  - `oddity_heuristic_listener` predicts points concentrated on unique singletons when duplicates are present, but strictly uniform allocation when all objects are distinct.
  - `competitor_confusion_speaker` predicts points concentrated on objects with the lowest exponential similarity to all other display objects.
  - `isolated_graded_costly_l3` predicts points proportional to minimum Hamming distance to the nearest neighbor (`object_isolation`).
  - `rsa_l2_singleton_feat_color_valence_l0` (and its family) predicts points increasing with both feature complexity (`feature_count`), total Hamming distance (`distinct`), and singleton status.
- **Synonyms (`synonyms: true`)**: When two features have the exact same extension across objects, recursive RSA models (`rsa_l1`, `rsa_l2`, `isolated_graded_costly_l3`, `rsa_l2_singleton_feat_color_valence_l0`) shift their speaker probabilities because the speaker has two synonymous messages available to describe objects bearing those features, changing the pragmatic competition against objects with unique features. By contrast, `oddity_heuristic_listener` and `lexical_preemption_heuristic` (when the synonymous features are shared across multiple objects) are insensitive to synonymous alternatives in ways that diverge from recursive pragmatic models.

---

## 3. What Each Theory Predicts in This Paradigm

1. **`competitor_confusion_speaker` (Live)**:
   - **Utterance trials**: Strongly penalizes referents that share features with visually similar competitors that also satisfy the heard word. Allocates high points to the referent that is most feature-distinct from the other objects satisfying $u$.
   - **Prior trials**: Allocates points toward isolated objects that have the largest Hamming distance from all other objects in the display.
2. **`isolated_graded_costly_l3` (Live)**:
   - **Utterance trials**: Depth-3 recursive amplification sharpens point allocations toward the pragmatically implicated target, modulated by graded semantic truth ($\gamma$) and extension cost (`cost_weight`).
   - **Prior trials**: Allocates points according to the minimum Hamming distance to any single nearest neighbor (`object_isolation`).
3. **`oddity_heuristic_listener` (Live)**:
   - **Utterance trials**: Allocates zero points to false objects; among true objects, shifts points toward unique singletons (`is_singleton = 1`) and away from identical duplicates (`beta > 0`), but allocates **uniformly** among all true objects whenever all true objects are singletons (even if one has a unique distinguishing feature that could have been named!).
   - **Prior trials**: Shifts points toward singletons in displays containing duplicates; completely uniform when all objects are distinct.
4. **`rsa_l2_singleton_feat_color_valence_l0` (Live)** & **`rsa_l2_singleton_distinct_feat_color_l0` / `rsa_l2_singleton_feat_color_familiar_l0` (Bar)**:
   - **Utterance trials**: Depth-2 pragmatic reasoning over a literal $L_0$, weighted by the rich perceptual prior (singleton bonus + total Hamming distinctiveness + feature count bonus). Allocates strong majority points to the depth-2 target while favoring feature-rich, distinct singletons.
   - **Prior trials**: Allocates points unevenly even when all objects are distinct singletons, favoring objects with more features (`w_features > 0`) and greater sum-of-Hamming-distances (`w_distinct > 0`).
5. **`rsa_l2` (Live)** & **`rsa_l1` (Live)**:
   - **Utterance trials**: Pure specificity implicatures driven by recursive Theory of Mind (`rsa_l2` resolves twice-implicated "complex" reference games and allocates more extreme points on depth-2 contrasts than `rsa_l1`).
   - **Prior trials**: Strictly uniform point allocation across all objects.
6. **`lexical_preemption_heuristic` (Bar)**:
   - **Utterance trials**: Penalizes candidate referents proportional to the number of unmentioned **uniquely distinguishing** features (`extension == 1`) they possess. Crucially, if an alternative feature is shared by 2 objects in a 4-object scene (`extension == 2`), `lexical_preemption_heuristic` assigns zero penalty (`is_dist == 0`), whereas `rsa_l1` and `rsa_l2` still compute a partial informativeness implicature!
   - **Prior trials**: Strictly uniform point allocation.
7. **`softmax_surprisal_isolated_costly_l4` (Bar)**:
   - **Utterance trials**: Depth-4 recursion with a power-law belief choice rule (`beta ~ 2.03`), producing near-categorical point allocations on utterance trials while weighting objects by contextual feature surprisal (`w_surprisal`) on prior trials.
8. **`literal_listener` (Bar)**:
   - **Utterance trials**: Splits the 10 points equally among all objects of which the heard word is literally true, with 0 points on false objects.
   - **Prior trials**: Splits the 10 points equally among all objects.
