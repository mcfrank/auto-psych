# Compression cost: judging randomness by how incompressible a sequence is

People judge a sequence as more random-looking when it cannot be compressed
into a short description. The index of randomness is the description cost of
the sequence's finest run-length encoding: sequences with very few runs (long
streaks, periodic structure) are highly compressible and judged non-random,
and so are sequences with the maximum possible number of runs (perfectly
alternating HT), because both admit a brief description. Genuinely random
sequences fall in between, resisting compression. A person rates the more
incompressible of the two sequences as the more random one.
