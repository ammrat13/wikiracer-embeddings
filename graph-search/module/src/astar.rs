//! # The A* Algorithm
//!
//! This module runs the A* algorithm. It takes its parameters "cooked", meaning
//! they have already been extracted from the Memgraph procedure arguments.

use rsmgp_sys::define_type;
use rsmgp_sys::memgraph::Memgraph;
use rsmgp_sys::path::Path;
use rsmgp_sys::result::{Result, ResultRecord};
use rsmgp_sys::rsmgp::{NamedType, Type};
use rsmgp_sys::vertex::Vertex;

use c_str_macro::c_str;

use crate::heuristic::Heuristic;

/// The output of the A* algorithm, in terms of Memgraph types. This corresponds
/// with the fields in `AStarResult` and `AStarStats`.
pub static ASTAR_OUTPUT_TYPES: &[NamedType] = &[
    define_type!("path", Type::Path),
    define_type!("expanded_nodes", Type::Int),
    define_type!("relaxed_edges", Type::Int),
    define_type!("heuristic_runs", Type::Int),
    define_type!("heuristic_nodes", Type::Int),
];

/// The result of running A*. Has the path, as well as statistics.
pub struct AStarResult {
    pub path: Path,
    pub stats: AStarStats,
}

impl AStarResult {
    /// Add the result of A* to a result record to be returned to the user. The
    /// result record must have fields corresponding to `OUTPUT_TYPES`.
    pub fn add_to(&self, result: &ResultRecord) -> Result<()> {
        result.insert_path(c_str!("path"), &self.path)?;
        self.stats.add_to(result)
    }
}

/// The statistics of running A*. Returned in the result.
pub struct AStarStats {
    /// How many times the inner loop of A* was run. I.e., how many times we
    /// looked at a node.
    pub expanded_nodes: usize,
    /// How many times we put a node in the priority queue. I.e., how many times
    /// an edge was successfully relaxed, opening up further exploration.
    pub relaxed_edges: usize,
    /// How many times the heuristic was run.
    pub heuristic_runs: usize,
    /// How many nodes the heuristic was evaluated on. This is not the same as
    /// `heuristic_runs`, as we batch calls to the heuristic.
    pub heuristic_nodes: usize,
}

impl AStarStats {
    fn add_to(&self, result: &ResultRecord) -> Result<()> {
        result.insert_int(c_str!("expanded_nodes"), self.expanded_nodes as i64)?;
        result.insert_int(c_str!("relaxed_edges"), self.relaxed_edges as i64)?;
        result.insert_int(c_str!("heuristic_runs"), self.heuristic_runs as i64)?;
        result.insert_int(c_str!("heuristic_nodes"), self.heuristic_nodes as i64)
    }
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
            heuristic_runs: 0,
            heuristic_nodes: 0,
        },
    });
}
