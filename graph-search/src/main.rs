mod heuristic;
mod io;

use argparse::ArgumentParser;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use yaml_rust::YamlLoader;

use heuristic::null::NullHeuristic;
use heuristic::torch::TorchHeuristic;
use heuristic::Heuristic;
use io::Graph;
use io::TestPair;

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

    // Read the configuration file
    let config = std::fs::read_to_string(config_path).unwrap();
    let config = YamlLoader::load_from_str(&config).unwrap();
    let config = config.get(0).unwrap();
    // Extract parameters
    let graph_path = config["data"]["graph"]["bson"].as_str().unwrap();
    let embeddings_path = config["data"]["embeddings"][1536].as_str().unwrap();
    let test_pairs_path = config["data"]["graph"]["test-set"].as_str().unwrap();

    // Read the graph
    let graph = std::fs::File::open(graph_path).expect("Could not open graph file");
    let graph: Graph = bson::from_reader(graph).expect("Could not parse graph file");
    // Read the embeddings
    let embeddings = std::fs::File::open(embeddings_path).expect("Could not open embeddings file");
    let embeddings = Array2::<f32>::read_npy(embeddings).expect("Could not parse embeddings file");
    // Create a CSV reader for the test pairs
    let test_pairs = std::fs::File::open(test_pairs_path).expect("Could not open test pairs");
    let mut test_pairs = csv::Reader::from_reader(test_pairs);

    // Construct the heuristic
    let heuristic: Box<dyn Heuristic> = match heuristic_name.as_str() {
        "null" => Box::new(NullHeuristic::create(heuristic_args)),
        "torch" => Box::new(TorchHeuristic::create(heuristic_args, &embeddings)),
        _ => panic!("Unknown heuristic: {}", heuristic_name),
    };

    // Create a CSV writer for the output
    let output = std::fs::File::create(output_path).expect("Could not create output file");
    let mut output = csv::Writer::from_writer(output);

    // Run A* on the test pairs
    for result in test_pairs.deserialize() {
        let result: TestPair = result.expect("Could not parse test pair");
        println!("{:?}", result);
    }
}
