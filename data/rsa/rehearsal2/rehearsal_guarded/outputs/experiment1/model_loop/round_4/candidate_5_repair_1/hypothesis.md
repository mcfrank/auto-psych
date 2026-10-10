Pragmatic listeners reason at depth 2 about speakers who avoid competitor confusion, penalize referring expressions based on the single most informative omitted alternative feature of the target, and account for visual singleton salience. When multiple descriptions apply to an object, speakers avoid using an ambiguous descriptor if the target possesses a superior, highly discriminative unmentioned alternative feature in the display, leading listeners to reject referents for which a better descriptor was withheld. On uninformative trials, listeners spontaneously favor unique visual singletons over duplicated items in the display.

Refining rsa_l2_singleton_confusion_omission by changing the functional form of omitted-alternative exhaustification from additive sum to peak alternative informativeness:
Speakers evaluate candidate referring expressions by penalizing expressions in proportion to the maximum informativeness among unmentioned alternative features of the intended referent, rather than summing informativeness across all unmentioned features. This improves fit on complex displays by properly differentiating referents with genuinely superior disambiguating alternative descriptors from cluttered objects that merely possess multiple non-discriminative background features, avoiding the spurious penalty accumulation that linear summation produces.

Differences from rsa_l2_singleton_confusion_omission:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In compute_exhaustification_violation, replaced the additive sum of unmentioned feature informativeness with the maximum informativeness across all unmentioned alternative features true of the referent (masking out the evaluated utterance and taking the maximum specificity over remaining true features).
