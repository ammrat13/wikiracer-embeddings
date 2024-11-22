//! # The Null Heuristic
//!
//! Return 0 if the target vertex is the same as the query vertex, and 1
//! otherwise.

use super::Heuristic;

use rsmgp_sys::result::Result;
use rsmgp_sys::vertex::Vertex;

pub struct NullHeuristic;

impl NullHeuristic {
    pub fn new() -> Self {
        NullHeuristic
    }
}

impl Heuristic for NullHeuristic {
    fn estimate(&self, target: &Vertex, queries: &[Vertex]) -> Result<Vec<f32>> {
        Ok(queries
            .iter()
            .map(|query| if query.id() == target.id() { 0.0 } else { 1.0 })
            .collect())
    }
}
