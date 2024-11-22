//! # Graph Search Evaluation Module
//!
//! Having trained models for distance estimation on Simple English Wikipedia,
//! we can use them for A*. This module provides a procedure to run A* on a
//! graph. It allows using a TensorFlow model for distance estimation, as well
//! as a "null" heuristic for baseline.

mod astar;
mod heuristic;

use rsmgp_sys::memgraph::Memgraph;
use rsmgp_sys::mgp::{mgp_graph, mgp_list, mgp_memory, mgp_module, mgp_result};
use rsmgp_sys::result::{Error, Result};
use rsmgp_sys::rsmgp::{set_memgraph_error_msg, NamedType, Type};
use rsmgp_sys::value::Value;
use rsmgp_sys::{close_module, define_procedure, define_type, init_module};

use c_str_macro::c_str;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

use astar::ASTAR_OUTPUT_TYPES;
use heuristic::null::NullHeuristic;
use heuristic::Heuristic;

close_module!(|| -> Result<()> { Ok(()) });

init_module!(|memgraph: &Memgraph| -> Result<()> {
    // Graph search procedure for the null heuristic.
    memgraph.add_read_procedure(
        graph_search_null,
        c_str!("graph_search_null"),
        &[
            define_type!("source", Type::Vertex),
            define_type!("target", Type::Vertex),
        ],
        &[],
        ASTAR_OUTPUT_TYPES,
    )?;

    Ok(())
});

define_procedure!(graph_search_null, |memgraph: &Memgraph| -> Result<()> {
    let heur = NullHeuristic::new();
    graph_search(memgraph, &heur)
});

/// Common graph search procedure for all heuristics. The client creates the
/// heuristic, then calls this procedure. The source and target vertices must be
/// at index 0 and 1 in the arguments, and the result record has a fixed schema.
fn graph_search(memgraph: &Memgraph, heur: &impl Heuristic) -> Result<()> {
    let result = memgraph.result_record()?;
    let args = memgraph.args()?;

    // Extract the source and target vertices.
    let source: Value = args.value_at(0)?.to_mgp_value(memgraph)?.to_value()?;
    let source = match source {
        Value::Vertex(v) => v,
        _ => return Err(Error::UnableToCopyVertex),
    };
    let target = args.value_at(1)?.to_mgp_value(memgraph)?.to_value()?;
    let target = match target {
        Value::Vertex(v) => v,
        _ => return Err(Error::UnableToCopyVertex),
    };

    // Run A* and get the result.
    let astar_result = astar::astar(memgraph, &source, &target, heur)?;
    astar_result.add_to(&result)?;

    Ok(())
}
