//! Module containing all the heuristics we can use for graph search.

pub mod null;
pub mod torch;

use crate::io::NodeIndex;

pub trait Heuristic {
    /// Given a target node and a list of query nodes, return the estimated
    /// distance from each query node to the target node. Keep the same order in
    /// the returned vector as in the input vector.
    fn estimate(&self, target: NodeIndex, queries: &[NodeIndex]) -> Vec<f32>;
}
