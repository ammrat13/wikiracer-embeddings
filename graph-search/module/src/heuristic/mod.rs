//! # Graph Search Heuristics
//!
//! This module provides heuristics that are used by A*. It provides a common
//! trait for them, as well as the null and PyTorch heuristics.

pub mod null;

use rsmgp_sys::vertex::Vertex;

pub trait Heuristic {
    /// Given a target vertex and a list of query vertices, return the estimated
    /// distance from each query vertex to the target vertex. The estimates are
    /// returned in the same order as the query vertices.
    fn estimate(&self, target: &Vertex, queries: &[&Vertex]) -> Vec<f32>;
}
