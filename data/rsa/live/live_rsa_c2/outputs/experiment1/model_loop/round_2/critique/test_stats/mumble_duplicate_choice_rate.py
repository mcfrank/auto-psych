# name: mumble_duplicate_choice_rate
# description: Proportion of choices for any duplicated object class among mumble trials containing both singleton and duplicated objects.
def test_statistic(df):
    unique_objs = df.loc[df['query'] == 'prior', 'objects'].unique()
    dup_map = {}
    for obj_s in unique_objs:
        objs = json.loads(obj_s)
        classes = {}
        for i, o in enumerate(objs):
            t = tuple(o)
            classes[t] = classes.get(t, 0) + 1
        dup_classes = {objs.index(list(t)) for t, cnt in classes.items() if cnt > 1}
        has_singletons = any(cnt == 1 for cnt in classes.values())
        if dup_classes and has_singletons:
            dup_map[obj_s] = dup_classes
    prior_rows = df[df['query'] == 'prior']
    targets = [dup_map.get(o, set()) for o in prior_rows['objects']]
    mask = [len(s) > 0 for s in targets]
    if not any(mask):
        return 0.0
    sub_df = prior_rows[mask]
    sub_targets = [targets[i] for i, m in enumerate(mask) if m]
    chosen = [c in s for c, s in zip(sub_df['choice'], sub_targets)]
    return float(np.mean(chosen))
