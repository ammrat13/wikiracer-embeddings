//! Structures that we read from files. Even though these are deserialized, they
//! are used throughout the application. For example, the `Graph` struct is used
//! in A*. Hence, we define them here instead of in `main.rs`.

use serde::Deserialize;

/// Everything we do deals with node indicies, which are just numbers.
pub type NodeIndex = usize;

/// The graph we collected from Memgraph.
#[derive(Debug, Deserialize)]
pub struct Graph {
    pub num_nodes: usize,
    pub adjacency_list: Vec<Vec<NodeIndex>>,
}

#[derive(Debug, Deserialize)]
pub struct TestPair {
    #[serde(rename = "source-index")]
    pub source: NodeIndex,
    #[serde(rename = "target-index")]
    pub target: NodeIndex,
}
