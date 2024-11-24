//! Return 0 if the query is the same as the target, otherwise return 1.

use crate::heuristic::Heuristic;
use crate::io::NodeIndex;

pub struct NullHeuristic;

impl Heuristic for NullHeuristic {
    fn estimate(&self, target: NodeIndex, queries: &[NodeIndex]) -> Vec<f32> {
        queries
            .iter()
            .map(|&query| if query == target { 0.0 } else { 1.0 })
            .collect()
    }
}

impl NullHeuristic {
    /// Parse arguments to create the heuristic. May exit.
    pub fn create(args: Vec<String>) -> NullHeuristic {
        {
            let mut ap = argparse::ArgumentParser::new();
            ap.set_description("Null heuristic");

            let res = ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr());
            if let Err(code) = res {
                std::process::exit(code);
            }
        }
        NullHeuristic
    }
}
