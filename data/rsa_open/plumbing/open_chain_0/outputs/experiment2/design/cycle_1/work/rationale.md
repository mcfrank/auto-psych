# Experiment 2 Rationale: Multiscale Referent Selection and Competitor Evaluation Across 2- to 6-Object Contexts

## 1. Purpose of Experiment 2: Testing Generalizability Beyond Small-Context Point Allocation

In Experiment 1 (`cand_bet`), participants completed a 10-point probability allocation (betting) task (`bet` response, `allocation` link) across displays of 2 to 4 objects (`[2,2]` through `[4,4]`). On held-out log predictive density (`1_fits.md`), two live theories finished in a statistical dead heat:
- **`rsa_l2_singleton_feat_color_valence_l0`** (best on Experiment 1)
- **`isolated_graded_costly_l3`** (`61.6 ± 117.2` nats behind the best — within 0.53 SE)
- **`competitor_confusion_speaker`** (`1356.6 ± 384.0` nats behind)

Following Yarkoni (2020, *"The generalizability crisis"*), a theory that achieves competitive fit on a single paradigm and context scale can do so by adapting flexible parameters to task-specific or scale-specific regularities rather than capturing the underlying cognitive mechanism. Inspection of the posterior parameter draws fit to data through Experiment 1 (`context/draws/*.npz`) reveals two critical confounds that allowed `isolated_graded_costly_l3` and `rsa_l2_singleton_feat_color_valence_l0` to mimic one another in Experiment 1:

1. **Collinearity of Nearest-Neighbor Isolation vs. Cumulative Hamming Distinctiveness at Small Context Sizes ($N \in \{2, 3, 4\}$):**
   - `rsa_l2_singleton_feat_color_valence_l0` defines contextual object salience using a discrete singleton indicator (`is_singleton`, fitted weight $w_{\text{singleton}} = 1.008 \pm 0.031$) plus **cumulative Hamming distance** to all other objects in the display ($\text{distinct}_i = \sum_{j \neq i} d_H(i, j)$, fitted weight $w_{\text{distinct}} = 0.226 \pm 0.008$).
   - `isolated_graded_costly_l3` defines object salience solely via **nearest-neighbor Hamming isolation** ($\text{isolation}_i = \min_{j \neq i} d_H(i, j)$, fitted weight $w_{\text{distinct}} = 0.527 \pm 0.012$).
   - Notice the exact ratio of the fitted distinctiveness weights: $0.527 / 0.226 = 2.33$. In Experiment 1, the mean display size was $N \approx 3.44$ objects, meaning each object had $N - 1 \approx 2.44$ competitors! Thus, at $N \in \{2, 3, 4\}$, multiplying the sum of distances to $N - 1 \approx 2.33$ competitors by $0.226$ produced nearly identical prior log-odds to multiplying the minimum distance by $0.527$.

2. **Compensatory Overfitting of Graded Truth ($\gamma < 0$) and Negative Utterance Extension Costs in `isolated_graded_costly_l3`:**
   - Although `isolated_graded_costly_l3` was formulated so that additional competing features diminish semantic applicability ($\text{precision}_r = \exp(-\gamma (\text{fc}_r - 1))$), its posterior draw on Experiment 1 inverted the sign of $\gamma$ to a large negative value ($\gamma = -2.945 \pm 0.390$) alongside a negative extension cost weight ($\text{cost\_weight} = -0.571 \pm 0.116$).
   - With $\gamma = -2.945$, each additional feature on an object multiplies its literal truth value in `graded_lex` by $\exp(2.945) \approx 19.0$ for real words while leaving the sink utterance fixed at $1.0$. In a 10-point allocation (`bet`) paradigm on small displays, this massive boost to multi-feature objects at $L_0$, combined with depth-3 recursion ($L_3$) and negative extension costs (favoring shared words), allowed `isolated_graded_costly_l3` to approximate the graded point splits observed in Experiment 1.

Experiment 2 is designed to break both confounds simultaneously by testing **cross-paradigm and cross-scale generalization** across displays spanning **2 to 6 objects** ($N \in \{2, 3, 4, 5, 6\}$).

---

## 2. Targeted Theoretical Disagreements

Our design targets three sharp, quantitative disagreements among the live theories (`rsa_l2_singleton_feat_color_valence_l0`, `isolated_graded_costly_l3`, `competitor_confusion_speaker`) and the baseline theories (`literal_listener`, `rsa_l1`, `rsa_l2`, `rsa_l1_salience`, `rsa_l1_shared_prior`, `lexical_preemption_heuristic`, `oddity_heuristic_listener`, `evaluative_prominence_l2`, `familiar_isolated_graded_costly_l2`, `softmax_surprisal_isolated_costly_l4`):

### Disagreement A: Global Crowd Contrast ($\sum_{j \neq i} d_H(i, j)$) vs. Local Nearest-Neighbor Distance ($\min_{j \neq i} d_H(i, j)$) vs. Exponential Similarity ($\sum_{j \neq i} e^{-d_H(i, j)}$)
When display size expands to $N = 5$ and $N = 6$ objects (`[5,3]`, `[6,3]`):
- **Crowd vs. Outlier Prior Trials (`heard = null`):** Consider a 6-object scene with four identical featureless base objects `[0,0,0]`, one 1-feature singleton `[0,0,1]`, and one 3-feature singleton `[1,1,1]`:
  - **`rsa_l2_singleton_feat_color_valence_l0`** gives both `[0,0,1]` and `[1,1,1]` the singleton bonus ($w_{\text{singleton}} = 1.008$), but their cumulative Hamming distances diverge sharply: `[1,1,1]` has $\sum_j d_H = 4(3) + 2 = 14$ ($0.226 \times 14 = 3.164$), whereas `[0,0,1]` has $\sum_j d_H = 4(1) + 2 = 6$ ($0.226 \times 6 = 1.356$), and each `[0,0,0]` has $\sum_j d_H = 4$ ($0.226 \times 4 = 0.904$, no singleton bonus). Consequently, `rsa_l2_singleton_feat_color_valence_l0` predicts a $\exp(3.164 - 1.356) \approx 6.1\times$ preference for `[1,1,1]` over `[0,0,1]` and a $26.3\times$ preference over each `[0,0,0]`.
  - **`isolated_graded_costly_l3`** evaluates only $\min_{j \neq i} d_H(i, j)$: `[1,1,1]` has $\min d_H = 2$ (to `[0,0,1]`), `[0,0,1]` has $\min d_H = 1$ (to `[0,0,0]`), and `[0,0,0]` has $\min d_H = 0$. With $w_{\text{distinct}} = 0.527$, `isolated_graded_costly_l3` predicts only a $\exp(0.527) \approx 1.69\times$ ratio between `[1,1,1]` and `[0,0,1]`, and is completely insensitive to whether there are two or четыре copies of `[0,0,0]`.
  - **`competitor_confusion_speaker`** penalizes baseline confusion $\sum_{j \neq i} \exp(-d_H(i, j))$ with $w_{\text{confusion}} = 0.786$: each `[0,0,0]` suffers massive self-crowd penalty ($3 \times e^0 + e^{-1} + e^{-3} = 3.42$), `[0,0,1]` has moderate similarity to the crowd ($4 e^{-1} + e^{-2} = 1.61$), and `[1,1,1]` has minimal similarity ($4 e^{-3} + e^{-2} = 0.33$).
  - **`oddity_heuristic_listener`** treats all singletons (`[0,0,1]` and `[1,1,1]`) as equally odd ($\text{is\_singleton} = 1$), predicting a tie between them.

### Disagreement B: Asymmetric Clone Clusters with Near Neighbors
Consider a 6-object scene where all objects have a near neighbor at Hamming distance 1 ($\min_{j \neq i} d_H(i, j) = 1$ for every singleton), or where a minority pair of clones (2 copies of `[0,0,1]`) competes against a majority cluster of clones (4 copies of `[1,1,0]`):
- Under **`isolated_graded_costly_l3`**, both the minority pair and the majority cluster have $\min_{j \neq i} d_H(i, j) = 0$, so their prior isolation is identical ($0.0$).
- Under **`rsa_l2_singleton_feat_color_valence_l0`**, both have $\text{is\_singleton} = 0$, but each member of the minority pair has $\sum_j d_H = 4 \times 3 = 12$, whereas each member of the majority cluster has $\sum_j d_H = 2 \times 3 = 6$. Thus `rsa_l2_singleton_feat_color_valence_l0` predicts a $\exp(0.226 \times (12 - 6)) = 3.88\times$ preference for each minority-pair object!

### Disagreement C: Pragmatic Specificity Implicature vs. Inverted Graded Truth ($\gamma = -2.945$) on Heard-Word Trials
When a speaker utters a shared word $u$ that is true of both a single-feature object (e.g., `[0,0,1]`) and one or more multi-feature objects (e.g., `[1,1,1]` or `[0,1,1]`):
- **`rsa_l2_singleton_feat_color_valence_l0`**, **`rsa_l1`**, **`rsa_l2`**, and **`lexical_preemption_heuristic`** predict standard **specificity / scalar implicature**: because the multi-feature object possesses alternative distinguishing features the speaker could have named, hearing the shared word favors the minimal 1-feature referent `[0,0,1]` (modulated in `rsa_l2_singleton_feat_color_valence_l0` by singleton status and cumulative distinctiveness).
- **`isolated_graded_costly_l3`** (with fitted $\gamma = -2.945$ and $\text{cost\_weight} = -0.571$) predicts distorted pragmatic dynamics across 1-, 2-, and 3-feature competitors because `graded_lex` weights real words by $\exp(+2.945 (\text{fc}_r - 1))$ ($19\times$ for 2 features, $361\times$ for 3 features) against a unit sink utterance ($1.0$), causing $L_0$ and odd/even recursion tiers to strongly skew toward feature-dense referents and shared extensions.
- **`competitor_confusion_speaker`** ($L_1$ with $\alpha = 0.183, w_{\text{confusion}} = 0.786$) has very weak pragmatic recursion ($\alpha \approx 0.18$) and instead selects referents where the heard word $u$ has minimal overlap with visually similar distractors ($\text{confusion}_{u,r} = \sum_{j \neq r} \text{lex}_{u,j} e^{-d_H(r,j)}$).

---

## 3. Why This Task, Scene Space, and Response Type

1. **Task & Response Type:**
   - Experiment 1 used 10-point betting (`bet` / `allocation`). Testing **listener referent choice (`choice`)** (and comparing against paired `choice + rating` in our exploration) evaluates whether theories fitted to continuous point allocations generalize to direct referent decisions under a distinct response link without fixed-sum budget artifacts.
   - Including both **heard-word trials** (`synonyms: true`, allowing multiple candidate words per object) and **uninformative prior trials** (`prior_queries: true`) jointly constrains the perceptual prior parameters ($w_{\text{singleton}}, w_{\text{distinct}}, w_{\text{confusion}}$) and the recursive pragmatic parameters ($\alpha, \gamma, \text{cost\_weight}$) while ensuring clean link parameter recovery (`temperature` and `lapse`).
2. **Scene Space (`[[2,2], [3,2], [3,3], [4,3], [5,3], [6,3]]` / multiscale 2–6 objects):**
   - Spanning 2 to 6 objects breaks the $N \approx 3.4$ scaling degeneracy between $\sum_{j \neq i} d_H(i, j)$ and $\min_{j \neq i} d_H(i, j)$, achieving high expected information gain (more than $2\times$ the closed-campaign floor per dollar) while maintaining fast, unambiguous visual scanning using the pilot-validated `robots` SVG domain (`antenna`, `bowtie`, `earmuffs`, `wheels`).

---

## 4. Summary of Predictions by Theory

| Trial Condition | `rsa_l2_singleton_feat_color_valence_l0` | `isolated_graded_costly_l3` | `competitor_confusion_speaker` | Heuristic / Baseline Models |
|---|---|---|---|---|
| **Prior trial ($N=6$): 4 clones `[0,0,0]` + singletons `[0,0,1]` & `[1,1,1]`** | Strong preference for `[1,1,1]` over `[0,0,1]` ($\sim 6.1\times$) due to cumulative $\sum_j d_H = 14$ vs. $6$, and both over `[0,0,0]` ($w_{\text{singleton}} = 1.01$). | Modest preference for `[1,1,1]` over `[0,0,1]` ($\sim 1.7\times$, $\min d_H = 2$ vs. $1$); insensitive to clone crowd size. | Strong avoidance of `[0,0,0]` clone crowd; prefers `[1,1,1]` over `[0,0,1]` via exponential similarity decay. | `oddity_heuristic`: equal preference for `[0,0,1]` and `[1,1,1]`; `literal`/`rsa_l1`/`rsa_l2`: uniform. |
| **Prior trial ($N=6$): 2 clones `[0,0,1]` vs. 4 clones `[1,1,0]`** | Strongly favors minority clones `[0,0,1]` ($\sum_j d_H = 12$ vs. $6$, $\sim 3.9\times$). | Predicts exact tie ($\min d_H = 0$ for all objects). | Strongly favors minority clones `[0,0,1]` (1 identical clone vs. 3 identical clones). | All uniform ($\text{is\_singleton} = 0$ for all objects). |
| **Heard shared word ($N=6$): singleton `[0,0,1]` vs. 3 clones of `[1,1,1]`** | Overwhelmingly selects `[0,0,1]` (combines depth-2 specificity implicature, singleton bonus, and higher $\sum_j d_H$). | Heavily distorted toward `[1,1,1]` by $\gamma = -2.945$ ($361\times$ graded-lex boost for 3-feature objects) and negative extension cost. | Selects `[0,0,1]` primarily via competitor-clone confusion penalty on `[1,1,1]` rather than high-$\alpha$ pragmatic recursion. | `lexical_preemption`: selects `[0,0,1]`; `literal_listener`: uniform across all 4 true referents ($25\%$ each). |
