# Experiment 2 Rationale: Probed Referent Likelihood Rating (`probed_referent_likelihood_rating`)

## 1. What This Experiment Is For and Why It Matters Next

Theories of pragmatic reference must generalize across task demands, response modalities, and structural scene complexity rather than overfitting to a single experimental format (Yarkoni, 2020). Across this study so far:
- **Closed campaign (Experiments 1–3)**: Tested **multi-object forced choice** (`choice` link), which collapses the listener's posterior distribution $P(r \mid u)$ to its mode/winner and masks graded credence on secondary candidate referents.
- **Open campaign Experiment 1 (`graded_belief_allocation`)**: Tested **simultaneous 10-point belief allocation** (`bet` / `allocation` link) across 3–4 objects and 2–3 features (`[[3, 2], [3, 3], [4, 3]]`). While Experiment 1 separated Depth-1 models from Depth-2 recursion (`rsa_l2_singleton_feat_color_valence_l0` lagged by $42.3 \pm 37.5$ nats), the top three live theories were statistically indistinguishable within ~1 standard error on held-out log predictive density:
  1. `bounded_depth_listener` (**best**)
  2. `rsa_l1_singleton_feat_color_valence_l0` ($5.1 \pm 17.2$ nats behind best)
  3. `bounded_depth_feature_parsimony` ($12.3 \pm 9.8$ nats behind best)

Why could Experiment 1 not decisively separate these three Depth-1 theories?
1. **Response-process confounding in simultaneous point splitting**: Splitting 10 integer points across all 3–4 objects simultaneously enforces an explicit sum-to-10 simplex constraint on every trial. Participants frequently rely on coarse integer-partition heuristics (e.g., `5-5`, `4-3-3`, or `10-0`), inflating Dirichlet-multinomial dispersion ($\kappa$) and obscuring fine-grained differences in the probability assigned to a specific critical competitor.
2. **Restricted structural scene space (`[3, 2], [3, 3], [4, 3]`)**: Experiment 1 included neither 4-feature scenes (`[3, 4], [4, 4]`), where targets and competitors vary across 0 to 4 features and create multi-tier lexical competition, nor 2-object minimal pairs (`[2, 2], [2, 3]`), nor 5-object displays (`[5, 2]`), where multiple duplicate groups of different sizes coexist.

**Experiment 2 (`probed_referent_likelihood_rating`)** resolves both limitations by introducing a **targeted single-referent evaluation paradigm (`rating` response type / ordered-probit `rating` link on a 1–7 scale)** across an expanded structural complexity space spanning **2 to 5 objects and 2 to 4 features** (`[[2, 2], [2, 3], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [5, 2]]`) with `prior_queries: true` and `synonyms: true`.

On each trial, a single candidate object (`probe`) is highlighted in an orange frame, and the participant rates how likely that specific object is to be the speaker's target from `1 (Definitely not)` to `7 (Definitely)`. Because the experimental design unit is a `(scene, probe)` pair ($5,275$ candidate units), greedy joint-EIG selection directly probes the exact referent in each display where the competing theories' latent posteriors diverge most strongly—without cross-object integer-budget coupling.

---

## 2. Theoretical Disagreements Targeted

### Disagreement 1: Discrete Singleton Salience Only vs. Multi-Cue Structural Priors at Depth 1
- **Theories**: `bounded_depth_listener` vs. `rsa_l1_singleton_feat_color_valence_l0`
- **Mechanism**: Both models operate at reasoning Depth 1 ($L_1$ inverting a speaker $S_1$ who simulates a literal listener $L_0$) and share the discrete singleton salience weight $w_{\text{singleton}}$. However:
  - `bounded_depth_listener` posits that the object prior depends *exclusively* on whether an object has an identical clone in the display ($\text{prior}(r) \propto \exp(w_{\text{singleton}} \cdot \text{is\_singleton}(r))$). Whenever all objects in a display are distinct ($\text{is\_singleton}(r) = 1$ for all $r$), its prior is strictly uniform ($1/N$), and among non-clone objects it assigns zero weight to visual feature complexity (`feature_count`) or continuous Hamming isolation (`distinct`).
  - `rsa_l1_singleton_feat_color_valence_l0` posits a shared multi-cue structural prior combining discrete singleton status, continuous Hamming distance to all other display objects ($\text{distinct}(r) = \sum_{r'} \|f(r) - f(r')\|_1$), and feature elaboration ($\text{feature\_count}(r)$).
- **How Experiment 2 separates them**:
  - In `[3, 4]` and `[4, 4]` scenes where all objects are unique singletons or where two non-clone candidates differ sharply in feature count (e.g., 1 feature vs. 4 features) or Hamming isolation (e.g., an isolated object at Hamming distance 3 from neighbors vs. a clustered object at Hamming distance 1), `bounded_depth_listener` predicts identical prior ratings and purely $S_1$-driven informative ratings, whereas `rsa_l1_singleton_feat_color_valence_l0` predicts systematic shifts governed by $w_{\text{distinct}}$ and $w_{\text{features}}$ across both prior and informative trials.
  - In `[5, 2]` scenes with asymmetric clone groups (e.g., 3 identical clones vs. 2 identical clones), `is_singleton` is $0$ for both groups, so `bounded_depth_listener` rates a member of the 2-clone group and a member of the 3-clone group identically on prior trials, whereas `distinct` in `rsa_l1_singleton_feat_color_valence_l0` differentiates minority clones from majority clones.

### Disagreement 2: Communicative Speaker Normalization vs. Heuristic Feature-Parsimony Penalty
- **Theories**: `bounded_depth_listener` vs. `bounded_depth_feature_parsimony`
- **Mechanism**: Both models use the exact same singleton-only prior on uninformative (`prior`) trials. On informative trials, however, `bounded_depth_feature_parsimony` multiplies the listener's posterior weight by an explicit feature-count penalty $\exp(-\beta \cdot \text{feature\_count}(r))$ ($\beta > 0$) *on top of* the speaker's utterance choice probability $P_{S1}(u \mid r)$.
- **How Experiment 2 separates them**:
  - Under `bounded_depth_listener`, an object with additional features is penalized on informative trials *only* to the extent that those additional features provide more informative alternative words (smaller extension $|u'|$) in $S_1$'s denominator $\sum_{u'} \text{lex}(u', r) |u'|^{-\alpha}$.
  - When an extra feature is **universally shared** across objects, has a **large extension** ($|u'| = 3$ or $4$ in `[4, 3]` and `[4, 4]`), or when **synonyms** manipulate the number of true words on the target vs. competitor, `bounded_depth_listener` and `bounded_depth_feature_parsimony` make sharply divergent predictions for the probed multi-feature referent's rating.
  - Furthermore, comparing ratings of high-feature-count objects on **prior trials** (where `bounded_depth_feature_parsimony` applies *no* $\beta$ penalty) vs. **informative trials** (where it applies $\exp(-\beta \cdot \text{feature\_count}(r))$) directly isolates $\beta$ from prior feature preferences ($w_{\text{features}}$).

### Disagreement 3: Bounded Depth 1 vs. Recursive Depth 2+ and Non-Bayesian Heuristics
- **Theories**: Depth-1 models (`bounded_depth_listener`, `rsa_l1_singleton_feat_color_valence_l0`, `bounded_depth_feature_parsimony`, `rsa_l1`, `rsa_l1_salience`, `rsa_l1_shared_prior`) vs. Depth-2/3/4 models (`rsa_l2_singleton_feat_color_valence_l0`, `rsa_l2`, `evaluative_prominence_l2`, `isolated_graded_costly_l3`, `familiar_isolated_graded_costly_l2`, `softmax_surprisal_isolated_costly_l4`) vs. Heuristic/Confusion models (`lexical_preemption_heuristic`, `oddity_heuristic_listener`, `competitor_confusion_speaker`, `literal_listener`).
- **How Experiment 2 separates them**:
  - In 3- and 4-feature reference chains (e.g., $O_1=\{A, B\}, O_2=\{B, C\}, O_3=\{C, D\}$), Depth-1 listeners ($L_1$) evaluate $S_1$ over literal semantics $L_0$, where features of equal extension size receive equal speaker weight; Depth-2 listeners ($L_2$) invert $S_2$ simulating $L_1$, propagating disambiguation along the chain and compounding the referent prior across two recursion steps.
  - Directly probing the scalar or chain competitor on a 7-point scale captures whether listeners assign moderate graded credence (as predicted by bounded Depth-1 RSA) or near-floor credence (as predicted by Depth-2+ over-implicature and `lexical_preemption_heuristic`).

---

## 3. Why This Task, Scene Space, and Response Type

1. **High EIG and 100% Identification of the Live Theories**:
   - In full EIG evaluation (`work/reports/eig_d59103b863d7.json`), `probed_referent_likelihood_rating` achieves **3.913 of 4.170 bits** ($0.02934\text{ bits/\$}$ at $\$133.33$, more than **2.28x the required floor** of $0.01287\text{ bits/\$}$, and outperforming the closed campaign reference paradigm's $3.432\text{ bits}$ / $0.02574\text{ bits/\$}$).
   - Crucially, across the three distinct live theories that were statistically tied in Experiment 1, this design achieves **100% identification accuracy (`p_correct = 1.00`, `mean_posterior_on_truth = 1.0000`)** on every single one:
     - `bounded_depth_listener`: `p_correct = 1.000`, `mean_posterior_on_truth = 1.0000`
     - `rsa_l1_singleton_feat_color_valence_l0`: `p_correct = 1.000`, `mean_posterior_on_truth = 1.0000`
     - `bounded_depth_feature_parsimony`: `p_correct = 1.000`, `mean_posterior_on_truth = 1.0000`
2. **Exact Parameter Identifiability and Recovery (`link-check`)**:
   - Including 4-feature scenes (`[3, 4], [4, 4]`) provides multi-extension utterance sets ($|u| \in \{1, 2, 3, 4\}$) that cleanly decouple speaker rationality $\alpha$ from link decision `temperature` and probit cutpoint `spread`.
   - In `link-check` (`work/reports/link-check_d59103b863d7.json`, 4,000 responses simulated from the 6-parameter live theory `rsa_l1_singleton_feat_color_valence_l0`), all 3 NUTS replicates converged with **0 misses across all 10 parameters** (`alpha: 0, w_singleton: 0, w_distinct: 0, w_features: 0, w_valence: 0, w_color: 0, center: 0, spread: 0, lapse: 0, temperature: 0`).
3. **Clear, Non-Overlapping 4-Feature SVG Materials**:
   - Both domains (`snowmen`: `hat`, `scarf`, `broom`, `balloon`; `houses`: `chimney`, `flag`, `tree`, `sun`) have complete, recognizable bases and four spatially separated, concrete, single-word features placed in distinct quadrants of the $100 \times 100$ canvas so that any subset of 0 to 4 features combines without visual occlusion.

---

## 4. What Each Theory Predicts

- **`bounded_depth_listener` (Live)**:
  - *Informative trials*: Ratings of the probed referent increase with $S_1(u \mid r) \propto \text{lex}(u, r)|u|^{-\alpha} / \sum_{u'} \text{lex}(u', r)|u'|^{-\alpha}$ weighted by discrete singleton status $\exp(w_{\text{singleton}} \cdot \text{is\_singleton}(r))$. Multi-feature competitors retain moderate ratings (no over-implicature), and adding universally shared or non-distinguishing features does not penalize a referent once speaker alternatives are matched.
  - *Prior trials*: Probed unique singletons receive high ratings ($w_{\text{singleton}} > 0$) relative to identical clones, but all unique objects receive equal ratings regardless of feature count (`feature_count`) or Hamming isolation (`distinct`).
- **`rsa_l1_singleton_feat_color_valence_l0` (Live)**:
  - *Informative & Prior trials*: In addition to Depth-1 speaker informativeness and singleton preference, ratings systematically track continuous Hamming isolation ($w_{\text{distinct}} \cdot \text{distinct}(r)$) and feature count ($w_{\text{features}} \cdot \text{feature\_count}(r)$), differentiating isolated vs. clustered unique objects and minority vs. majority clone groups in `[5, 2]`, `[3, 4]`, and `[4, 4]` displays.
- **`bounded_depth_feature_parsimony` (Live)**:
  - *Informative trials*: Strong penalty $\exp(-\beta \cdot \text{feature\_count}(r))$ ($\beta > 0$) against probed objects with more visual features, suppressing ratings for 3- and 4-feature candidates even when extra features are shared background features or synonyms.
  - *Prior trials*: Reverts to the singleton-only prior ($\beta$ is inactive on prior trials), creating a sharp interaction between trial type (prior vs. informative) and object feature count.
- **`rsa_l2_singleton_feat_color_valence_l0` (Live — and its plain-scene equivalents `rsa_l2_singleton_distinct_feat_color_l0`, `rsa_l2_singleton_feat_color_familiar_l0`)**:
  - *Informative trials*: Depth-2 recursion ($L_2$ inverting $S_2$ simulating $L_1$) compounds the structural prior across two reasoning levels and produces strong second-order implicatures in multi-step reference chains (`[3, 3], [3, 4], [4, 4]`), driving ratings of scalar competitors toward the floor while strongly amplifying singleton/distinctiveness biases.
  - *Prior trials*: Follows the shared structural prior over singleton status, Hamming distinctiveness, and feature count.
- **Baseline and Earlier Reference Theories**:
  - `literal_listener`: Rates all literally true referents equally high (above midpoint) and all literally false referents at the floor (`1`), with uniform ratings on prior trials.
  - `rsa_l1` / `rsa_l2`: Purely truth-conditional Depth-1 and Depth-2 pragmatic inference with strictly uniform ratings on prior trials and zero sensitivity to clone vs. singleton status.
  - `rsa_l1_salience` / `rsa_l1_shared_prior`: Weight referents by feature count (`w_features`) without singleton or Hamming-distance sensitivity; `rsa_l1_shared_prior` enters the prior into $L_0$ as well as $S_1$.
  - `isolated_graded_costly_l3` / `familiar_isolated_graded_costly_l2` / `softmax_surprisal_isolated_costly_l4`: Predict graded truth decay ($\exp(-\gamma(\text{fc}-1))$) against multi-feature objects, extension-based utterance costs, and minimum-Hamming-distance isolation priors across Depth 2–4 recursion.
  - `evaluative_prominence_l2`: Depth-2 inference driven solely by feature-count prominence (`w_base * feature_count`), ignoring singleton status and Hamming isolation.
  - `lexical_preemption_heuristic`: Penalizes probed candidates solely by the count of unmentioned *uniquely distinguishing* (extension-1) features they possess, predicting no implicature when alternative features are shared by 2+ objects and uniform ratings on prior trials.
  - `competitor_confusion_speaker`: Penalizes words shared with feature-similar distractors ($\exp(-\text{dist})$), favoring probed referents that are visually dissimilar from other holders of the heard word.
  - `oddity_heuristic_listener`: Non-recursive heuristic that rates true referents solely by whether they are unique singletons vs. clones, ignoring alternative words on the objects.
