# Experiment 3 Design Rationale (`chain_0`): `multi_competitor_referent_choice`

## 1. What This Experiment Is For

Theories of pragmatic reference must generalize across task demands, response modalities, and contextual complexity (Yarkoni, 2020; Grice, 1975; Goodman & Frank, 2016). In `chain_0` so far, the theories have been tested on two graded listener tasks:
1. **Experiment 1 (`graded_belief_allocation`)**: 10-point belief allocation (`bet` / `allocation` link) across small 3–4 object, 2–3 feature displays (`sizes: [[3, 2], [3, 3], [4, 3]]`, $N=120 \times 15 = 1{,}800$ trials).
2. **Experiment 2 (`probed_referent_likelihood_rating`)**: 7-point Likert likelihood ratings of a single highlighted candidate referent (`rating` / `rating` link) across 2–5 objects and 2–4 features ($N=200 \times 20 = 4{,}000$ single-object ratings).

Across both earlier paradigms, **`rsa_l1_singleton_feat_color_valence_l0`** (depth-1 RSA with a multi-cue shared object prior combining discrete singleton salience, continuous Hamming distinctiveness, and visual feature complexity) ranked first, but three competing bounded- and recursive-depth theories remained statistically close on held-out log predictive density:
- **`bounded_depth_listener`** ($8.6 \pm 9.8$ nats behind on Exp 1; $0.1 \pm 20.3$ nats behind on Exp 2);
- **`bounded_depth_feature_parsimony`** ($10.0 \pm 10.2$ nats behind on Exp 1; $9.8 \pm 23.9$ nats behind on Exp 2);
- **`rsa_l2_singleton_feat_color_valence_l0`** ($29.4 \pm 21.3$ nats behind on Exp 1; $9.1 \pm 62.5$ nats behind on Exp 2);
- alongside **`feature_uncertainty_listener_2`** ($36.9 \pm 20.0$ nats behind on Exp 1; $41.0 \pm 46.9$ nats behind on Exp 2).

Experiment 2's single-object Likert ratings produced large clustered standard errors ($\pm 20.3$ to $\pm 62.5$ nats) because each trial probed only one object in isolation on an ordinal scale with cutpoint nuisance parameters (`center`, `spread`), rather than forcing participants to resolve direct winner-take-all competition across all candidate referents in the display. Meanwhile, Experiment 1's allocation task restricted the display space to at most 4 objects and 3 features ($N=120$, 1,800 trials), where continuous Hamming distinctiveness (`distinct`) and feature complexity (`feature_count`) vary over a narrow numerical range.

**`multi_competitor_referent_choice`** (`multi_competitor_referent_choice.json`) is designed to resolve these theoretical disagreements and test cross-paradigm generalization from graded ratings and point splits to **discrete winner-take-all referent selection (`choice` / `choice` link)** across an expanded scene space spanning **2 to 6 objects** and **2 to 4 features** (`sizes: [[2, 2], [2, 3], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [5, 2], [6, 2]]`, with `prior_queries: true` and `synonyms: true`), collecting **5,600 test choices** ($N = 200$ participants $\times$ $28$ test trials across $D = 56$ EIG-optimal scenes, yielding **100 responses per scene**).

---

## 2. Which Disagreements Among the Theories It Targets

In open-campaign scenes (where `grayscale = 0`, `familiarization = 0`, and `valence = 0`), the five live theories make sharply diverging structural predictions along four axes that our 2–6 object, 2–4 feature scene space directly targets:

### Disagreement A: Multi-Cue Structural Prior (`distinct` + `feature_count`) vs. Pure Singleton Salience (`is_singleton`)
- **`rsa_l1_singleton_feat_color_valence_l0`** and **`rsa_l2_singleton_feat_color_valence_l0`** define the object prior as:
  $$P_{\text{prior}}(r) \propto \exp\left(w_{\text{singleton}} \cdot \text{is\_singleton}[r] + w_{\text{distinct}} \cdot \text{distinct}[r] + w_{\text{features}} \cdot \text{feature\_count}[r]\right)$$
  where $\text{distinct}[r] = \sum_{r' \neq r} \sum_{u \notin \text{sink}} |\text{lex}[u, r] - \text{lex}[u, r']|$ is the total Hamming distance of object $r$ from all other objects across all real utterances.
- **`bounded_depth_listener`**, **`bounded_depth_feature_parsimony`**, and **`feature_uncertainty_listener_2`** define the object prior solely over binary singleton status:
  $$P_{\text{prior}}(r) \propto \exp\left(w_{\text{singleton}} \cdot \text{is\_singleton}[r]\right)$$
- **Why 4-feature and 5–6 object displays separate them**:
  In 2-object scenes, $\text{distinct}[0] = \text{distinct}[1]$ identically; in 3-object scenes, $\Delta\text{distinct}$ is at most 2–4 units. Given the posterior mean $w_{\text{distinct}} \approx 0.047\text{–}0.096$, small displays produce only subtle log-odds shifts. By contrast, in 4-feature (`[3, 4], [4, 4]`) and 5–6 object (`[5, 2], [6, 2]`) scenes where several objects cluster in feature space while one singleton is Hamming-isolated (or where two unique singletons have different feature counts and Hamming distances from duplicated clones, e.g., `[0,0,0,0], [0,0,0,0], [0,0,0,1], [1,1,1,1]`), $\Delta\text{distinct}$ reaches **6 to 12 Hamming units**.
  - On **prior trials** (`heard = null`) where multiple objects are unique singletons (`is_singleton = 1`), `bounded_depth_listener`, `bounded_depth_feature_parsimony`, and `feature_uncertainty_listener_2` predict **strictly equal choice probabilities** among all singletons, whereas `rsa_l1_singleton_feat_color_valence_l0` and `rsa_l2_singleton_feat_color_valence_l0` predict systematic preferences for the singleton with higher Hamming isolation (`distinct`) and/or feature elaboration (`feature_count`).
  - On **informative trials** where two singletons share the heard word and have equal speaker likelihood $P_{S_1}(u \mid r)$, `bounded_depth_listener` predicts a 50/50 split between them, whereas `rsa_l1_singleton_feat_color_valence_l0` favors the more Hamming-distinctive referent.

### Disagreement B: Shared Feature-Complexity Prior vs. Informative-Only Feature Parsimony (`beta`)
- **`bounded_depth_feature_parsimony`** introduces a strictly positive parsimony penalty $\beta \sim \text{LogNormal}(0, 1)$ that scales listener choice weights by $\exp(-\beta \cdot \text{feature\_count}[r])$ **only on informative trials**, while leaving prior trials governed solely by `is_singleton`:
  $$P_{L_1}^{\text{parsimony}}(r \mid u) \propto \exp\left(w_{\text{singleton}} \cdot \text{is\_singleton}[r] - \beta \cdot \text{feature\_count}[r]\right) P_{S_1}(u \mid r)$$
- **`rsa_l1_singleton_feat_color_valence_l0`** places `w_features` inside the shared prior affecting **both** prior and informative trials equally, while **`bounded_depth_listener`** has no feature-count penalty beyond standard speaker informativeness $P_{S_1}(u \mid r)$.
- **Why 4-feature displays and singleton-vs-duplicate trade-offs separate them**:
  In scenes where a minimal referent is duplicated (`is_singleton = 0`, e.g., two `{A}` objects) and competes against a unique multi-feature singleton (`is_singleton = 1`, e.g., `{A, B, C, D}`) when `"A"` is heard, singleton salience ($w_{\text{singleton}} \approx 1.76$) pulls toward `{A, B, C, D}` while speaker informativeness and feature parsimony pull toward `{A}`. Across 2-, 3-, and 4-feature competitors, `bounded_depth_feature_parsimony` predicts an exponential penalty $\exp(-k\beta)$ against $k$-feature competitors on informative trials (even when the extra features are synonyms or shared) paired with **zero** feature-count effect on prior trials.

### Disagreement C: Bounded Depth 1 (`L1`) vs. Recursive Depth 2 (`L2`)
- **`rsa_l1_singleton_feat_color_valence_l0`** inverts a depth-1 speaker $S_1(u \mid r) \propto \text{lex}[u, r] L_0(r \mid u)^\alpha$ where $L_0$ is purely truth-conditional (uniform over true referents). Thus the object prior enters **once** in the listener's posterior: $P_{L_1}(r \mid u) \propto \text{prior}(r) S_1(u \mid r)$.
- **`rsa_l2_singleton_feat_color_valence_l0`** inverts a depth-2 speaker $S_2(u \mid r) \propto \text{lex}[u, r] L_1(r \mid u)^\alpha$. Because $L_1(r \mid u)$ already incorporates $\text{prior}(r)$, `L2` **compounds the object prior across recursion tiers** ($\text{prior}(r)^{1+\alpha}$ before utterance normalization) and generates higher-order scalar implicatures in overlapping 3- and 4-feature chains (e.g., `{A}`, `{A, B}`, `{B, C}`, `{A, B, C, D}`).

### Disagreement D: Truth-Conditional Literal Semantics (`L0`) vs. Feature-Bundle Uncertainty (`lex / feature_count`)
- **`feature_uncertainty_listener_2`** normalizes literal compatibility in $L_0$ by each object's total feature count: $\text{lex\_norm}[u, r] = \text{lex}[u, r] / \max(\text{feature\_count}[r], 1)$.
- Consequently, even when an additional feature on an object is **universally shared** across all referents in the display (or when a speaker has no informative alternative word), `feature_uncertainty_listener_2` penalizes multi-feature referents inside $L_0$ itself by $1/k$, whereas all four truth-conditional $L_0$ theories predict no penalty for universally shared background features unless driven by `w_features` or `beta`.

---

## 3. Why This Task and Response Type (`choice` via `choice` Link)

We selected discrete forced-choice referent selection (`type: "choice"`, `link: "choice"`) over the 10-shape scene space `[[2, 2], [2, 3], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [5, 2], [6, 2]]` for four quantitative and methodological reasons:

1. **Cross-Paradigm Generalization in `chain_0`**:
   Experiment 1 tested graded multi-object point allocation (`bet` / `allocation`) and Experiment 2 tested single-object Likert ratings (`rating` / `rating`). Testing discrete forced-choice (`choice`) in Experiment 3 completes the triad of listener decision paradigms—from simplex allocation to single-item ordinal evaluation to multi-alternative winner-take-all choice—while extending the stimulus space to **6-object displays (`[6, 2]`)** and **4-feature displays (`[3, 4], [4, 4]`)**.
2. **Eliminating Single-Probe Information Loss and Scale Nuisance Variance**:
   In single-referent paradigms (`rating` and `slider`), each trial queries only one highlighted object out of $N \in \{2,\dots,6\}$ objects and requires estimating ordinal/beta cutpoint parameters (`center`, `spread` or `cut_low`, `cut_gap`, `phi`). In `choice`, every trial directly observes the outcome of competition across **all $N$ objects in the scene simultaneously**, and the `choice` link requires only two parameters (`lapse` and `temperature`).
3. **Why Speaker Links (`production`, `binary_judgment`) Are Inferior for the Current Live Set**:
   Because none of the five live theories define a separate `speaker` function, the harness derives speaker probabilities as $P_S(u \mid r) = P_L(r \mid u) / \sum_{u'} P_L(r \mid u')$. For a fixed target object $r$, any object-level factor ($\text{prior}(r)$ or $\exp(-\beta \cdot \text{feature\_count}[r])$) cancels out of the numerator and enters only weakly at second order through the listener normalization denominator, while excluding `prior_queries`. Exploration confirmed that `binary_judgment` (`eig --quick`: $3.070$ bits, $p_{\text{correct}} = 0.605$) and `production` (`eig --quick`: $3.335$ bits, $p_{\text{correct}} = 0.770$) yield substantially lower information and poorer recovery of `w_features` than `choice`.
4. **High Sample Size, High EIG, and Exact Parameter Recovery**:
   Because forced-choice trials take only ~4.5 seconds per trial, $N = 200$ participants can complete **28 test trials** (+ 2 catch + 1 practice = 31 trials total) in an estimated **3.4 minutes** against a declared **3.0 minutes** (within 13% of declared time, well inside the 30% pilot tolerance) at $\$0.50$/participant ($\$133.33$ total cost). This yields **5,600 total test choices** across $D = 56$ distinct trial units (**100 responses per scene**):
   - **Full `eig` (`work/reports/eig_3982e7ef618c.json`)**: **3.491 of 4.087 bits** ($p_{\text{correct}} = 0.792$ overall across all 17 context models, vs. $3.263$ bits and $p_{\text{correct}} = 0.714$ for the closed-campaign reference), achieving **0.02618 bits/\$** against a floor of **0.01224 bits/\$** (**214% of the floor**).
   - Among the five live theories, full `eig` achieves $p_{\text{correct}} = 0.966$ ($\text{mean posterior} = 0.930$) on `rsa_l1_singleton_feat_color_valence_l0`, $0.938$ ($0.921$) on `feature_uncertainty_listener_2`, $0.767$ ($0.635$) on `bounded_depth_listener`, $0.700$ ($0.674$) on `bounded_depth_feature_parsimony`, and $0.625$ ($0.419$) on `rsa_l2_singleton_feat_color_valence_l0`.
   - **Full `link-check` (`work/reports/link-check_3982e7ef618c.json`)**: `converged = True`, `recovered = True`, with **0 misses across all 8 parameters** (`alpha: 0, w_singleton: 0, w_distinct: 0, w_features: 0, w_valence: 0, w_color: 0, lapse: 0, temperature: 0`).

---

## 4. What Each Live Theory Predicts

1. **`rsa_l1_singleton_feat_color_valence_l0` (Incumbent)**:
   - **Prior trials**: Strong preference for unique singletons (`w_singleton` $\approx 1.76$) modulated by continuous Hamming distinctiveness (`w_distinct` $\approx 0.05\text{–}0.10$ per Hamming unit) and feature count (`w_features`). When two objects are both unique singletons in a 4–6 object display (e.g., `[0,0,0,0], [0,0,0,0], [0,0,0,1], [1,1,1,1]`), listeners will systematically choose the singleton with greater Hamming distance from the duplicate cluster (`[1,1,1,1]`) rather than choosing uniformly between the two singletons.
   - **Informative trials**: Bounded depth-1 pragmatic inference (`L1` over truth-conditional `L0`) combined multiplicatively with the 3-cue shared prior. Preserves moderate choice share on multi-feature scalar competitors when they are unique singletons or have high Hamming distinctiveness relative to the display.

2. **`bounded_depth_listener`**:
   - **Prior trials**: Choices depend **exclusively** on binary singleton status (`is_singleton`). Across all unique singletons in a display—regardless of whether an object has 0, 1, or 4 features or sits far from a cluster in Hamming space—choice probabilities are **strictly uniform**.
   - **Informative trials**: Depth-1 RSA (`L1` over truth-conditional `L0`) weighted only by `is_singleton`. Any two singletons with identical literal speaker probabilities $P_{S_1}(u \mid r)$ must be chosen at identical rates regardless of their Hamming isolation or feature count.

3. **`bounded_depth_feature_parsimony`**:
   - **Prior trials**: Identical to `bounded_depth_listener`—choices depend solely on binary `is_singleton`, with **zero** sensitivity to `feature_count` or `distinct`.
   - **Informative trials**: Applies an explicit exponential penalty $\exp(-\beta \cdot \text{feature\_count}[r])$ ($\beta > 0$) against candidate referents with additional features, above and beyond speaker informativeness $P_{S_1}(u \mid r)$. In scenes pitting a duplicated minimal referent against a unique 3- or 4-feature singleton, or comparing candidate referents that share extra background features or synonyms, `bounded_depth_feature_parsimony` predicts a much steeper suppression of multi-feature referents than `bounded_depth_listener` or `rsa_l1_singleton_feat_color_valence_l0`.

4. **`rsa_l2_singleton_feat_color_valence_l0`**:
   - **Prior trials**: Follows the same 3-cue prior (`is_singleton`, `distinct`, `feature_count`) as `rsa_l1_singleton_feat_color_valence_l0`.
   - **Informative trials**: Because the depth-2 listener `L2` inverts a speaker `S2` who simulates `L1` (which already contains the prior), the object prior is **compounded across recursion tiers** ($\text{prior}(r)^{1+\alpha}$), and scalar implicatures in 3- and 4-feature scenes are sharper and exhibit second-order cascade effects across overlapping feature chains.

5. **`feature_uncertainty_listener_2`**:
   - **Prior trials**: Governed solely by binary `is_singleton`.
   - **Informative trials**: Because literal semantics $L_0$ divides compatibility by each object's feature count ($1/\text{feature\_count}[r]$), multi-feature objects are strongly penalized even when their extra features are universally shared across all objects in the scene or have synonymous labels that do not create speaker alternatives under truth-conditional $L_0$.
