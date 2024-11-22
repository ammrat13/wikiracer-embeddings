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

use heuristic::null::NullHeuristic;
use heuristic::Heuristic;

close_module!(|| -> Result<()> { Ok(()) });

init_module!(|memgraph: &Memgraph| -> Result<()> {
    // We have multiple graph search procedures to accomodate the different
    // heuristics. They have a lot in common though. We define the common output
    // types here. We unfortunately can't define common input types since types
    // can't be cloned.
    let graph_search_output = &[
        define_type!("path", Type::Path),
        define_type!("expanded_nodes", Type::Int),
        define_type!("relaxed_edges", Type::Int),
    ];

    // Graph search procedure for the null heuristic.
    memgraph.add_read_procedure(
        graph_search_null,
        c_str!("graph_search_null"),
        &[
            define_type!("source", Type::Vertex),
            define_type!("target", Type::Vertex),
        ],
        &[],
        graph_search_output,
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

    // Run A*.
    let astar_result = astar::astar(memgraph, &source, &target, heur)?;

    // Populate the result record.
    result.insert_path(c_str!("path"), &astar_result.path)?;
    result.insert_int(
        c_str!("expanded_nodes"),
        astar_result.stats.expanded_nodes as i64,
    )?;
    result.insert_int(
        c_str!("relaxed_edges"),
        astar_result.stats.relaxed_edges as i64,
    )?;

    Ok(())
}
