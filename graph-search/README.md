# Graph Search Evaluation

This folder contains code to run A* on the graph with a given heuristic. The
"Null" heuristic is used as a control, but PyTorch models are used for as the
treatment group. The program outputs statistics on the path that was eventually
found, as well as how long A* had to run to find it.

In addition to A*, there is also the greedy walk benchmark. In it, steps are
taken according to what the heuristic believes the closest node to the target
is. The figure of merit for that is how often the target node can be found
before the cutoff.

Auxiliary collection and evaluation scripts are in the `scripts/` directory.
Notably, this program operates on a local copy of the graph instead of the one
in Memgraph. Originally, the plan was to build a Memgraph plugin, but it appears
PyTorch does not work inside a query module. Hence, there is a script to
download and serialize the graph.

## Graph Search Results

Models with a hidden length of `128, 192` and a maximum distance of `4, 5, 6`
were tested. All of them blow the Null heuristic out of the water in terms of
the number of nodes expanded, but they are about 20x slower. A hidden length of
`192` performs 15-20% better than `128`, reflective of its 10% lower loss. A
maximum distance of 5 seems to perform best, but the effect of maximum distance
on nodes explored is comparatively weak - at most 11%.

## Greedy Walk Results

Usually, the heuristics can succeed 15-20% of the time. There's a lot of noise
in the data, so it's hard to glean more than that. In general, it seems a larger
hidden length performs better, and a lower maximum distance also performs
better.
