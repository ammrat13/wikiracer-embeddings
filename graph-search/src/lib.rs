use rsmgp_sys::memgraph::Memgraph;
use rsmgp_sys::mgp::{mgp_graph, mgp_list, mgp_module, mgp_memory, mgp_result};
use rsmgp_sys::result::MgpResult;
use rsmgp_sys::rsmgp::{Type, NamedType, set_memgraph_error_msg};
use rsmgp_sys::{close_module, define_procedure, define_type, init_module};

use c_str_macro::c_str;
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;

init_module!(|memgraph: &Memgraph| -> MgpResult<()> {
    memgraph.add_read_procedure(
        hello_world,
        c_str!("hello_world"),
        &[],
        &[],
        &[define_type!("output_string", Type::String)],
    )?;
    Ok(())
});

define_procedure!(hello_world, |memgraph: &Memgraph| -> MgpResult<()> {
    let result = memgraph.result_record()?;
    result.insert_string(
        c_str!("output_string"),
        c_str!("Hello, World!"),
    )?;
    Ok(())
});

close_module!(|| -> MgpResult<()> {
    Ok(())
});
