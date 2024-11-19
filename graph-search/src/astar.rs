//! # The A* Algorithm
//!
//! This module runs the A* algorithm. It takes its parameters "cooked", meaning
//! they have already been extracted from the Memgraph procedure arguments.

use rsmgp_sys::memgraph::Memgraph;
use rsmgp_sys::path::Path;
use rsmgp_sys::result::Result;
use rsmgp_sys::vertex::Vertex;

use crate::heuristic::Heuristic;

/// The result of running A*. Has the path, as well as statistics.
pub struct AStarResult {
    pub path: Path,
    pub stats: AStarStats,
}

/// The statistics of running A*. Returned in the result.
pub struct AStarStats {
    pub expanded_nodes: usize,
    pub relaxed_edges: usize,
}

pub fn astar(
    memgraph: &Memgraph,
    source: &Vertex,
    target: &Vertex,
    heur: &impl Heuristic,
) -> Result<AStarResult> {
    return Ok(AStarResult {
        path: Path::make_with_start(source, memgraph)?,
        stats: AStarStats {
            expanded_nodes: 0,
            relaxed_edges: 0,
        },
    });
}
