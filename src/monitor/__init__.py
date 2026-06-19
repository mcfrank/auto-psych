"""Live monitor for in-progress human studies.

While a study collects data on Prolific + Firebase, this module surfaces what is
happening in real time: how many participants have submitted, how their data
looks, and the recruitment status on Prolific. Its first duty is to catch
degenerate data early — the kind of silent failure (e.g. every participant
choosing the same side) that wasted an earlier pilot.
"""
