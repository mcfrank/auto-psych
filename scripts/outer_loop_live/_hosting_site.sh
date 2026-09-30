#!/bin/bash
# The Firebase Hosting site of one parallel live run, shared by
# start_full_run.sh (which prints the sites before the typed "yes") and
# submit_parallel.sh (which deploys each run to its own).
#
#   hosting_site <firebase project> <run_label> <run>
#     -> <project>-<run_label>-<run>   e.g. auto-psych-2c5da-sr-oct26-run1
#     -> <project>-<run>               with an empty run_label
#
# The run_label keeps one series of runs from reusing (and replacing the pages
# of) an earlier series' sites in the same project. Firebase refuses a site ID
# over 30 characters or with anything but lowercase letters, digits and '-';
# such a name stops the launch here, before anything is copied or submitted,
# not at the deploy after the page has been built.
hosting_site() {
  local project="$1" series="$2" run="$3" site
  site="$(echo "${project}${series:+-$series}-${run}" | tr '[:upper:]' '[:lower:]')"
  if [[ ! "$site" =~ ^[a-z0-9-]+$ ]] || (( ${#site} > 30 )); then
    echo "Hosting site '$site' (${#site} characters) is not a valid Firebase Hosting site ID:" \
      "at most 30 characters, only lowercase letters, digits and '-'. Shorten run_label or change its characters." >&2
    return 1
  fi
  echo "$site"
}
