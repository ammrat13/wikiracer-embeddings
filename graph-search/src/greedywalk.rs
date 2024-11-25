//! Greedy walk benchmark for distance estimation heuristics. We start at the
//! source node and take the edge with the smallest estimated distance to the
//! target. We continue this process until we reach the target or the walk
//! cutoff.

use core::f32;

use serde::Serialize;

use crate::heuristic::Heuristic;
use crate::io::{Graph, NodeIndex};

#[derive(Debug, Serialize)]
pub struct GreedyWalkResult {
    /// How many steps the walk took before reaching the target, or `None` if
    /// the walk was cut off.
    #[serde(rename = "steps")]
    pub steps: Option<usize>,
}

/// Perform a greedy walk from the source node to the target node using the
/// given heuristic. The walk will be cut off after `walk_cutoff` steps.
pub fn greedy_walk(
    graph: &Graph,
    heur: &Box<dyn Heuristic>,
    source: &NodeIndex,
    target: &NodeIndex,
    cutoff: usize,
) -> GreedyWalkResult {
    let mut steps_count = 0usize;
    let mut cur = *source;

    while cur != *target {
        // If we took too long, give up. Doing this here means that the returned
        // count will be less than or equal to the cutoff.
        if steps_count >= cutoff {
            return GreedyWalkResult { steps: None };
        }

        // Get the neighbors of the current node
        let neighbors = graph
            .adjacency_list
            .get(cur)
            .expect("Node index too large")
            .clone();

        // Run the heuristic on all neighbors
        let heuristics = heur.estimate(*target, &neighbors);

        // Find the neighbor with the smallest estimated distance
        let mut min_heur = f32::INFINITY;
        let mut min_i = None::<usize>;
        for (i, h) in heuristics.iter().enumerate() {
            if h < &min_heur {
                min_heur = *h;
                min_i = Some(i);
            }
        }

        // If we didn't find a neighbor, we're stuck
        let min_i = match min_i {
            Some(i) => i,
            None => return GreedyWalkResult { steps: None },
        };

        // Move to the neighbor
        cur = *neighbors.get(min_i).unwrap();
        steps_count += 1;
    }

    // We broke out of the loop, so we reached the target
    GreedyWalkResult {
        steps: Some(steps_count),
    }
}
