//! Run the A* algorithm on a graph using the specified heuristic. Return the
//! distance found between the source and target nodes, as well as statistics.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use serde::Serialize;

use crate::{
    heuristic::Heuristic,
    io::{Graph, NodeIndex},
};

#[derive(Debug, Serialize)]
pub struct AStarResult {
    /// The distance found between the source and target nodes.
    #[serde(rename = "distance")]
    pub distance: Option<usize>,

    /// How many nodes were dequeued from the priority queue.
    #[serde(rename = "stat-nodes-expanded")]
    pub nodes_expanded: usize,
    /// How many nodes were enqueued into the priority queue.
    #[serde(rename = "stat-nodes-generated")]
    pub nodes_generated: usize,

    /// How many times the heuristic was called.
    #[serde(rename = "stat-heuristic-calls")]
    pub heuristic_calls: usize,
    /// How many nodes the heuristic was called on. This is not equal to the
    /// number of calls since we do batching.
    #[serde(rename = "stat-heuristic-nodes")]
    pub heuristic_nodes: usize,

    // How long the search took, in seconds.
    #[serde(rename = "time-seconds")]
    pub time: f64,
}

impl AStarResult {
    fn new() -> AStarResult {
        AStarResult {
            distance: None,
            nodes_expanded: 0,
            nodes_generated: 0,
            heuristic_calls: 0,
            heuristic_nodes: 0,
            time: 0.0,
        }
    }
}

/// An entry in the priority queue for A*. Contains a node and its cost. Note
/// that the costs are compared in reverse order, so that the priority queue
/// returns the smallest cost first. We also remember the G score of the node so
/// we can skip it if needed.
struct PQEntry {
    node_id: NodeIndex,
    fscore: f32,
    gscore: usize,
}

impl PartialEq for PQEntry {
    fn eq(&self, other: &Self) -> bool {
        self.fscore == other.fscore
    }
}

impl Eq for PQEntry {}

impl PartialOrd for PQEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PQEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Safety: We never deal with Infity or NaN, so we can make a total
        // order on the floats.
        other.fscore.total_cmp(&self.fscore)
    }
}

/// Entry for each vertex in the array of true distances. We don't reconstruct
/// the path, so we only need the G score.
struct GSEntry {
    gscore: usize,
}

pub fn astar(
    graph: &Graph,
    heur: &Box<dyn Heuristic>,
    source: &NodeIndex,
    target: &NodeIndex,
) -> AStarResult {
    // Keep track of statistics
    let mut ret = AStarResult::new();
    let start_time = std::time::Instant::now();

    let mut pq: BinaryHeap<PQEntry> = BinaryHeap::new();
    let mut gs: HashMap<NodeIndex, GSEntry> = HashMap::new();
    // We'll cache the heuristic values for each vertex so we don't have to keep
    // calling it.
    let mut heur_cache: HashMap<NodeIndex, f32> = HashMap::new();

    // Add the source vertex. It doesn't matter what the heuristic value is, as
    // we immediately pop it off the priority queue.
    pq.push(PQEntry {
        node_id: *source,
        fscore: f32::NEG_INFINITY,
        gscore: 0,
    });
    gs.insert(*source, GSEntry { gscore: 0 });

    // The main loop of A*
    while let Some(PQEntry {
        node_id: cur_id,
        gscore: cur_queued_gscore,
        ..
    }) = pq.pop()
    {
        // Safety: Anything that makes it into the priority queue must have a
        // metadata entry. This is because we update the G scores before adding
        // stuff to the priority queue.
        let cur_gsentry = gs.get(&cur_id).unwrap();
        let cur_gscore = cur_gsentry.gscore;
        ret.nodes_expanded += 1;

        // The gscore should never increase.
        debug_assert!(cur_queued_gscore >= cur_gscore);
        // If we popped outdated information, skip it.
        if cur_queued_gscore > cur_gscore {
            continue;
        }

        // If we reached the target, we're done.
        if cur_id == *target {
            ret.distance = Some(cur_gscore);
            ret.time = start_time.elapsed().as_secs_f64();
            return ret;
        }

        // Get a list of neighbors that have a shorter path to them through the
        // current node.
        let to_relax = graph
            .adjacency_list
            .get(cur_id)
            .unwrap()
            .iter()
            .map(|n| *n)
            .filter(|n| {
                let to_cur_gscore = cur_gscore + 1;
                if let Some(GSEntry { gscore: to_prev_gscore }) = gs.get(n) {
                    *to_prev_gscore > to_cur_gscore
                } else {
                    true
                }
            }).collect::<Vec<NodeIndex>>();

        // Find the nodes in that list for which we don't have a heuristic
        // value.
        let to_heur = to_relax
            .iter()
            .filter(|n| !heur_cache.contains_key(n))
            .map(|n| *n)
            .collect::<Vec<NodeIndex>>();
        if !to_heur.is_empty() {
            let heur_values = heur.estimate(*target, &to_heur);
            for (n, h) in to_heur.iter().zip(heur_values.iter()) {
                heur_cache.insert(*n, *h);
            }
            ret.heuristic_calls += 1;
            ret.heuristic_nodes += to_heur.len();
        }

        // Relax the neighbors
        for n in to_relax {
            let to_cur_gscore = cur_gscore + 1;
            let to_cur_fscore = to_cur_gscore as f32 + heur_cache.get(&n).unwrap();
            debug_assert!(match gs.get(&n) {
                Some(GSEntry { gscore: to_prev_gscore }) => *to_prev_gscore > to_cur_gscore,
                None => true,
            });
            pq.push(PQEntry {
                node_id: n,
                fscore: to_cur_fscore,
                gscore: to_cur_gscore,
            });
            gs.insert(n, GSEntry { gscore: to_cur_gscore });
            ret.nodes_generated += 1;
        }
    }

    // We couldn't find a path
    ret.time = start_time.elapsed().as_secs_f64();
    ret
}
