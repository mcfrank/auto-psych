#!/bin/bash
# On jagupard28: pack nov26b-jag28's run3/bigelow_piantadosi_mixture cell and its agent tree,
# without fit caches, into home, for the move to tiger5 (preempted twice by jag-standard, 2026-11-01).
set -euo pipefail
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-jag28
id=f8fb9bba6e6fa0a9
D=/sailhome/wurgaft/auto-psych-runs/move_jag28_run3
[[ $(cat $W/run3/bigelow_piantadosi_mixture/agent_tree_id) == "$id" ]]
mkdir -p $D
tar -C $W -cf $D/cell.tar --exclude='*.nc' run3/bigelow_piantadosi_mixture
tar -C /scr/wurgaft/auto-psych/agent_trees -cf $D/tree.tar --exclude='*.nc' --exclude=mcmc_cache $id
ls -la $D
