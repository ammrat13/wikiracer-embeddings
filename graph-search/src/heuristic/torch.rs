//! Heuristic based on a PyTorch model. The model must return one less than the
//! true distance for all queries in the batch.

use crate::heuristic::Heuristic;
use crate::io::NodeIndex;

use ndarray::Array2;
use tch::jit::CModule;
use tch::{Device, IndexOp, Kind, Tensor};

/// The longest possible embedding length.
const MAX_EMBEDDING_LENGTH: usize = 1536;

pub struct TorchHeuristic {
    model: CModule,
    device: Device,
    embeddings: Tensor,
}

impl Heuristic for TorchHeuristic {
    fn estimate(&self, target: NodeIndex, queries: &[NodeIndex]) -> Vec<f32> {
        let n_q: i64 = queries.len().try_into().unwrap();
        let n_l: i64 = *self.embeddings.size().get(1).unwrap();

        let queries_i64 = queries
            .iter()
            .map(|&q| q.try_into().unwrap())
            .collect::<Vec<i64>>();

        let t = self
            .embeddings
            .i(target as i64)
            .expand(&[n_q, n_l], false)
            .to(self.device);
        let q = self.embeddings.index_select(
            0,
            &Tensor::from_slice(&queries_i64)
                .to_kind(Kind::Int64)
                .to(self.device),
        );

        let res = self.model.forward_ts(&[t, q]).expect("Failed to run model");
        let res = res.to_device(Device::Cpu);
        let res = res.clamp_min(0.0);
        let res = res + 1.0;

        Vec::<f32>::try_from(res).expect("Failed to read output")
    }
}

impl TorchHeuristic {
    /// Parse arguments to create the heuristic. May exit.
    pub fn create(args: Vec<String>, embedding_data: &Array2<f32>) -> TorchHeuristic {
        let mut model_path = String::from("model.pt");
        let mut embedding_length = MAX_EMBEDDING_LENGTH;
        {
            let mut ap = argparse::ArgumentParser::new();
            ap.set_description("Torch heuristic");
            ap.refer(&mut embedding_length)
                .metavar("EMBEDDING_LENGTH")
                .add_option(
                    &["-l", "--embedding-length"],
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
        if embedding_length > MAX_EMBEDDING_LENGTH {
            eprintln!("Embedding length is too large");
            std::process::exit(1);
        }

        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(&model_path, device).expect("Could not load model");

        // Convert the NumPy array to a PyTorch tensor
        let embeddings = embedding_data
            .as_slice()
            .expect("Bad format for embeddings");
        let embeddings = Tensor::from_slice(embeddings)
            .view([embedding_data.nrows() as i64, embedding_data.ncols() as i64])
            .to_kind(Kind::Float);
        // Truncate the embeddings to the desired length, and renormalize them
        let embeddings = embeddings.slice(
            1,
            0,
            TryInto::<i64>::try_into(embedding_length).expect("Embedding length too large"),
            1,
        );
        let embeddings_norm = Tensor::einsum("ni,ni->n", &[&embeddings, &embeddings], None::<i64>)
            .sqrt()
            .unsqueeze(1);
        let embeddings = embeddings / embeddings_norm;
        // Finally, move the embeddings to the device
        let embeddings = embeddings.to_device(device);

        TorchHeuristic {
            model,
            device,
            embeddings,
        }
    }
}
