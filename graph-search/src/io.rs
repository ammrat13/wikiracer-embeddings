//! Structures that we read from files. Even though these are deserialized, they
//! are used throughout the application. For example, the `Graph` struct is used
//! in A*. Hence, we define them here instead of in `main.rs`.
//!
//! Additionally, this module provides a function to build all the internal
//! structures from command-line arguments. This is common to all applications,
//! so we define it here.

use std::fs::File;

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use yaml_rust::YamlLoader;

use crate::heuristic::null::NullHeuristic;
use crate::heuristic::torch::TorchHeuristic;
use crate::heuristic::Heuristic;

/// Everything we do deals with node indicies, which are just numbers.
pub type NodeIndex = usize;

/// The graph we collected from Memgraph.
#[derive(Debug, Deserialize)]
pub struct Graph {
    #[serde(rename = "adjacency_list")]
    pub adjacency_list: Vec<Vec<NodeIndex>>,
}

/// A single pair of nodes collected from Memgraph
#[derive(Debug, Deserialize)]
pub struct TestPair {
    #[serde(rename = "source-index")]
    pub source: NodeIndex,
    #[serde(rename = "target-index")]
    pub target: NodeIndex,
}

/// The arguments we read from the command line, parsed into a more usable form.
pub struct EvaluatedArgs {
    pub graph: Graph,
    pub heuristic: Box<dyn Heuristic>,
    pub test_pairs: csv::Reader<std::fs::File>,
}

/// Parse the command-line arguments and build the internal structures.
pub fn eval_args(
    config_path: &str,
    heuristic_name: &str,
    heuristic_args: Vec<String>,
) -> EvaluatedArgs {
    // Read the configuration file
    let config = std::fs::read_to_string(config_path).unwrap();
    let config = YamlLoader::load_from_str(&config).unwrap();
    let config = config.get(0).unwrap();
    // Extract parameters
    let graph_path = config["data"]["graph"]["bson"].as_str().unwrap();
    let embeddings_path = config["data"]["embeddings"][1536].as_str().unwrap();
    let test_pairs_path = config["data"]["graph"]["test-set"].as_str().unwrap();

    // Read the graph
    let graph = File::open(graph_path).expect("Could not open graph file");
    let graph: Graph = bson::from_reader(graph).expect("Could not parse graph file");
    // Read the embeddings
    let embeddings = File::open(embeddings_path).expect("Could not open embeddings file");
    let embeddings = Array2::<f32>::read_npy(embeddings).expect("Could not parse embeddings file");
    // Create a CSV reader for the test pairs
    let test_pairs = File::open(test_pairs_path).expect("Could not open test pairs");
    let test_pairs = csv::Reader::from_reader(test_pairs);

    // Construct the heuristic
    let heuristic: Box<dyn Heuristic> = match heuristic_name {
        "null" => Box::new(NullHeuristic::create(heuristic_args)),
        "torch" => Box::new(TorchHeuristic::create(heuristic_args, &embeddings)),
        _ => panic!("Unknown heuristic: {}", heuristic_name),
    };

    EvaluatedArgs {
        graph,
        heuristic,
        test_pairs,
    }
}
