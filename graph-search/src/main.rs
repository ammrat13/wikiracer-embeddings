mod astar;
mod heuristic;
mod io;

use std::fs::File;

use argparse::ArgumentParser;

use io::{EvaluatedArgs, TestPair};

fn main() {
    // Handle argument parsing
    // See: https://crates.io/crates/argparse
    let mut config_path = String::from("config.json");
    let mut output_path = String::from("results.csv");
    let mut heuristic_name = String::from("null");
    let mut heuristic_args = Vec::<String>::new();
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Graph search benchmark for distance estimation heuristics");
        ap.refer(&mut config_path).metavar("CONFIG").add_option(
            &["-c", "--config"],
            argparse::Store,
            "Path to the configuration file",
        );
        ap.refer(&mut output_path).metavar("OUTPUT").add_option(
            &["-o", "--output"],
            argparse::Store,
            "Path to the output file",
        );
        ap.refer(&mut heuristic_name)
            .metavar("HEURISTIC")
            .required()
            .add_argument("heuristic-name", argparse::Store, "Which heuristic to use");
        ap.refer(&mut heuristic_args)
            .metavar("HEURISTIC_ARGS")
            .add_argument(
                "heuristic-args",
                argparse::List,
                "Arguments for the heuristic",
            );
        ap.stop_on_first_argument(true);
        ap.parse_args_or_exit();
    }
    heuristic_args.insert(0, heuristic_name.clone());

    // Parse the arguments and build the internal structures
    let EvaluatedArgs {
        graph,
        heuristic,
        mut test_pairs,
    } = io::eval_args(&config_path, &heuristic_name, heuristic_args);

    // Create a CSV writer for the output
    let output = File::create(output_path).expect("Could not create output file");
    let mut output = csv::Writer::from_writer(output);

    // Run A* on the test pairs
    eprintln!("Running A* on test pairs...");
    for (i, tp) in test_pairs.deserialize().enumerate() {
        let tp: TestPair = tp.expect("Could not parse test pair");
        let res = astar::astar(&graph, &heuristic, &tp.source, &tp.target);
        output.serialize(res).expect("Could not write result");
        println!("Processed test pair {}", i + 1);
    }
}
