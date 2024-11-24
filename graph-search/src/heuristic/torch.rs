//! Heuristic based on a PyTorch model. The model must return one less than the
//! true distance for all queries in the batch.

use crate::heuristic::Heuristic;
use crate::io::NodeIndex;

use ndarray::Array2;
use tch::jit::CModule;
use tch::{Device, Kind, Tensor};

pub struct TorchHeuristic {
    model: CModule,
    device: Device,
    embeddings: Tensor,
}

impl Heuristic for TorchHeuristic {
    fn estimate(&self, target: NodeIndex, queries: &[NodeIndex]) -> Vec<f64> {
        todo!();
    }
}

impl TorchHeuristic {
    /// Parse arguments to create the heuristic. May exit.
    pub fn create(args: Vec<String>, embedding_data: &Array2<f32>) -> TorchHeuristic {
        let mut model_path: String = String::from("model.pt");
        let mut embedding_length: usize = 1536;
        {
            let mut ap = argparse::ArgumentParser::new();
            ap.set_description("Torch heuristic");
            ap.refer(&mut embedding_length)
                .metavar("EMBEDDING_LENGTH")
                .add_option(
                    &["-e", "--embedding-length"],
                    argparse::Store,
                    "Length of the embeddings the model expects",
                );
            ap.refer(&mut model_path)
                .metavar("MODEL")
                .required()
                .add_argument("model", argparse::Store, "Path to the PyTorch model");

            let res = ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr());
            if let Err(code) = res {
                std::process::exit(code);
            }
        }

        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(&model_path, device).expect("Could not load model");

        let embeddings = embedding_data
            .as_slice()
            .expect("Bad format for embeddings");
        let embeddings = Tensor::from_slice(embeddings)
            .view([embedding_data.nrows() as i64, embedding_data.ncols() as i64])
            .to_kind(Kind::Float)
            .to_device(device);

        TorchHeuristic {
            model,
            device,
            embeddings,
        }
    }
}
