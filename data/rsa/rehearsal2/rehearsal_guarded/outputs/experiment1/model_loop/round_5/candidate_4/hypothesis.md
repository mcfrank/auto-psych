Pragmatic listeners reason at depth 2 about speakers who avoid competitor confusion, penalize unmentioned informative features of the target, penalize broadly shared descriptors with utterance extension costs, and account for visual singleton salience and contextual feature distinctiveness. When multiple descriptions apply to an object, speakers prefer specific, less widely shared words over descriptors that extend to many scene objects, while avoiding ambiguous words when the target possesses superior unstated descriptors. On uninformative trials, listeners spontaneously favor items that contrast more strongly against the ensemble of display objects in overall feature space as well as discrete visual singletons.

Refining rsa_l2_singleton_confusion_omission_cost by incorporating contextual distinctiveness from rsa_l2_salience_confusion_l0:
Visual salience in reference games operates continuously across feature space rather than solely as an all-or-nothing binary distinction between duplicates and non-duplicates. Listeners and simulated speakers evaluate candidate referents with common-ground visual priors that combine discrete singleton salience with continuous contextual distinctiveness (the accumulated visual feature contrast between each object and all other display items), directing attention toward contextually distinctive objects even when all items in the display are distinct.

Differences from rsa_l2_singleton_confusion_omission_cost:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_distinct (Normal(0.0, 1.0), weighting the contextual feature distinctiveness prior over candidate referents).
- Parameters removed: None.
- Other terms: Added distinct calculation (accumulated pairwise visual feature distance between each referent and all other display objects); incorporated + params["w_distinct"] * distinct into the softmax object prior passed to simulated speakers S1 and S2 and governing uninformative prior trials.
