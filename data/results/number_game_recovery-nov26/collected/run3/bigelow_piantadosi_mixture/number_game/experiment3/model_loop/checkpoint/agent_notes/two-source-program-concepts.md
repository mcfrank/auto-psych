---
name: two-source-program-concepts
description: exp2 iter_4 c0: strays come from one secondary source per set (glitch or a 2nd rule) and the 2nd rule IS generalised from; MAP ties incumbent, RMSE .018, 2 s smoke
metadata:
  type: project
---
Exp2 iter_4/candidate_0 (2026-11-01), two_source_program_concepts. Each set has a main concept c1 (rules lambda_0/45, intervals /100^2, Beta reliability) and one secondary source: uniform glitches (1-mix) or a rule c2 (mix/45). "Yes" if t∈c1, or c2 produced ≥1 example and t∈c2. 4 params. MAP ll is -2033.0, against the incumbent familiar_strays_equal_rule_prior's -2032.3, with mix=0.16. RMSE from the incumbent is 0.018. Biggest differences are on targets that are the outlier itself or members of its rule (4,66,78,96→4/16/64).
Speed: all param-free sums go into prepare_observed as E[set,class,j,type] and Y[cell,class,j,type]. There's a G-subset enumeration per set (w1[c1,G]·w2[c2,G], and "neither contains t" via (1-mem).T @ w). The graph is just a weighted sum, so the smoke fit takes 2 s.
Related: [[familiar-strays-family-level]], [[equal-rule-prior-parsimony]], [[stray-generator-variants]]
