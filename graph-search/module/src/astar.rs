//! # The A* Algorithm
//!
//! This module runs the A* algorithm. It takes its parameters "cooked", meaning
//! they have already been extracted from the Memgraph procedure arguments.

use rsmgp_sys::define_type;
use rsmgp_sys::edge::Edge;
use rsmgp_sys::memgraph::Memgraph;
use rsmgp_sys::path::Path;
use rsmgp_sys::result::{Result, ResultRecord};
use rsmgp_sys::rsmgp::{NamedType, Type};
use rsmgp_sys::vertex::Vertex;

use c_str_macro::c_str;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::heuristic::Heuristic;

/// The output of the A* algorithm, in terms of Memgraph types. This corresponds
/// with the fields in `AStarResult` and `AStarStats`.
pub static ASTAR_OUTPUT_TYPES: &[NamedType] = &[
    define_type!("path", Type::Nullable, Type::Path),
    define_type!("expanded_nodes", Type::Int),
    define_type!("relaxed_edges", Type::Int),
    define_type!("heuristic_runs", Type::Int),
    define_type!("heuristic_nodes", Type::Int),
];

/// The result of running A*. Has the path, as well as statistics. The path may
/// not exist if the target is unreachable.
pub struct AStarResult {
    pub path: Option<Path>,
    pub stats: AStarStats,
}

impl AStarResult {
    /// Add the result of A* to a result record to be returned to the user. The
    /// result record must have fields corresponding to `OUTPUT_TYPES`.
    pub fn add_to(&self, result: &ResultRecord) -> Result<()> {
        match &self.path {
            Some(path) => result.insert_path(c_str!("path"), path)?,
            None => result.insert_null(c_str!("path"))?,
        }
        self.stats.add_to(result)
    }
}

/// The statistics of running A*. Returned in the result.
pub struct AStarStats {
    /// How many times the inner loop of A* was run. I.e., how many times we
    /// looked at a node.
    pub expanded_nodes: usize,
    /// How many times we put a node in the priority queue. I.e., how many times
    /// an edge was successfully relaxed, opening up further exploration.
    pub relaxed_edges: usize,
    /// How many times the heuristic was run.
    pub heuristic_runs: usize,
    /// How many nodes the heuristic was evaluated on. This is not the same as
    /// `heuristic_runs`, as we batch calls to the heuristic.
    pub heuristic_nodes: usize,
}

impl AStarStats {
    /// Return zero-initialized statistics counters.
    fn new() -> Self {
        AStarStats {
            expanded_nodes: 0,
            relaxed_edges: 0,
            heuristic_runs: 0,
            heuristic_nodes: 0,
        }
    }

    /// Add all the statistics to a result record to be returned to the user.
    fn add_to(&self, result: &ResultRecord) -> Result<()> {
        result.insert_int(c_str!("expanded_nodes"), self.expanded_nodes as i64)?;
        result.insert_int(c_str!("relaxed_edges"), self.relaxed_edges as i64)?;
        result.insert_int(c_str!("heuristic_runs"), self.heuristic_runs as i64)?;
        result.insert_int(c_str!("heuristic_nodes"), self.heuristic_nodes as i64)
    }
}

/// The type of a node ID. We deal in IDs because it seems like copying vertices
/// is not provided by the API.
type NodeId = i64;

/// An entry in the priority queue for A*. Contains a node and its cost. Note
/// that the costs are compared in reverse order, so that the priority queue
/// returns the smallest cost first.
struct PQEntry {
    node_id: NodeId,
    fscore: f32,
}

impl PartialEq for PQEntry {
    fn eq(&self, other: &Self) -> bool {
        self.fscore == other.fscore
    }
}

impl Eq for PQEntry {}

impl PartialOrd for PQEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PQEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Safety: We never deal with Infity or NaN, so we can make a total
        // order on the floats.
        other.fscore.total_cmp(&self.fscore)
    }
}

/// Entry for each vertex in the array of true distances. Also contains the edge
/// of our parent in the shortest path tree, though the start vertex has none.
struct GSEntry {
    gscore: usize,
    edge: Option<Edge>,
}

pub fn astar(
    memgraph: &Memgraph,
    source: &Vertex,
    target: &Vertex,
    heur: &impl Heuristic,
) -> Result<AStarResult> {
    // Keep track of stats for the entire execution.
    let mut stats = AStarStats::new();

    let mut pq: BinaryHeap<PQEntry> = BinaryHeap::new();
    let mut gs: HashMap<NodeId, GSEntry> = HashMap::new();
    // We'll cache the heuristic values for each vertex so we don't have to keep
    // calling it.
    let mut heur_cache: HashMap<NodeId, f32> = HashMap::new();

    // Evaluate the heuristic on the source vertex. Note that we have to do some
    // finagling to get the vertex out of the borrow.
    //
    // Safety: The heuristic will return a list of the same length as the query
    // list, so we can get the zeroth element.
    let heur_source = *heur
        .estimate(target, &[memgraph.vertex_by_id(source.id())?])?
        .get(0)
        .unwrap();
    // Add the source vertex.
    pq.push(PQEntry {
        node_id: source.id(),
        fscore: heur_source,
    });
    gs.insert(
        source.id(),
        GSEntry {
            gscore: 0,
            edge: None,
        },
    );
    heur_cache.insert(source.id(), heur_source);

    // Debug state
    let mut last_fscore = f32::NEG_INFINITY;

    // The main loop of A*.
    while let Some(PQEntry {
        node_id: cur_id,
        fscore: cur_fscore,
    }) = pq.pop()
    {
        // Safety: Anything that makes it into the priority queue must have a
        // metadata entry. This is because we update the G scores before adding
        // stuff to the priority queue.
        let cur_gsentry = gs.get(&cur_id).unwrap();
        let cur_gscore = cur_gsentry.gscore;
        stats.expanded_nodes += 1;

        // The fscores should always increase.
        debug_assert!(cur_fscore >= last_fscore);
        last_fscore = cur_fscore;

        // If we reached the target, we're done.
        if cur_id == target.id() {
            // Reconstruct the path. We can only construct in the forward
            // direction. So, we first collect all the edges in reverse order,
            // then reverse them.
            //
            // Note that all the vertices in the path are guaranteed to be in
            // the shortest path tree, so we don't need to check for that.

            let mut recon_edges = Vec::new();
            let mut recon_cur_id = cur_id;
            let mut recon_to_go = cur_gscore;

            loop {
                // Safety: Any explored node has a G score.
                let recon_cur_gsentry = gs.get(&recon_cur_id).unwrap();
                match &recon_cur_gsentry.edge {
                    Some(edge) => {
                        debug_assert!(recon_cur_id == edge.to_vertex()?.id());
                        debug_assert!(recon_cur_id != source.id());
                        debug_assert!(recon_to_go != 0);
                        recon_edges.push(edge);
                        recon_cur_id = edge.from_vertex()?.id();
                    }
                    None => {
                        debug_assert!(recon_cur_id == source.id());
                        debug_assert!(recon_to_go == 0);
                        break;
                    }
                }
                debug_assert!(recon_to_go > 0);
                recon_to_go -= 1;
            }

            let recon_path = Path::make_with_start(source, memgraph)?;
            for e in recon_edges.iter().rev() {
                recon_path.expand(e)?;
            }

            return Ok(AStarResult {
                path: Some(recon_path),
                stats,
            });
        }

        // Get a list of all the edges we need to relax.
        let edges_to_relax =
            memgraph
                .vertex_by_id(cur_id)?
                .out_edges()?
                .filter_map(|e| {
                    // Try to get the target vertex. If we can't, flag the
                    // error. Don't drop the error, though.
                    let to_id = match e.to_vertex() {
                        Ok(v) => v.id(),
                        Err(e) => return Some(Err(e)),
                    };
                    // Keep the edge if we can get to the target in a shorter
                    // way. If we don't have a gscore for the target, treat it
                    // as infinity.
                    let to_gscore = cur_gscore + 1;
                    match gs.get(&to_id) {
                        None => Some(Ok(e)),
                        Some(entry) => {
                            if to_gscore < entry.gscore {
                                Some(Ok(e))
                            } else {
                                None
                            }
                        }
                    }
                }).collect::<Result<Vec<Edge>>>()?;
        // If we don't have any edges to relax, we're done with this node.
        if edges_to_relax.is_empty() {
            continue;
        }

        // The nodes to relax are the target vertices of the edges.
        let node_ids_to_relax = edges_to_relax
            .iter()
            .map(|e| match e.to_vertex() {
                Ok(v) => Ok(v.id()),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<NodeId>>>()?;

        // First, update the G scores.
        for edge in edges_to_relax {
            let to_id = edge.to_vertex()?.id();
            let to_gscore = cur_gscore + 1;
            debug_assert!(match gs.get(&to_id) {
                None => true,
                Some(entry) => to_gscore < entry.gscore,
            });
            gs.insert(
                to_id,
                GSEntry {
                    gscore: to_gscore,
                    edge: Some(edge),
                },
            );
            stats.relaxed_edges += 1;
        }

        // Next, evaluate the heuristic on all the nodes to relax that don't
        // already have an entry in the heuristic cache. If we don't have to
        // evaluate the heuristic, we don't.
        let heur_queries = node_ids_to_relax
            .iter()
            .filter(|id| !heur_cache.contains_key(id))
            .map(|id| memgraph.vertex_by_id(*id))
            .collect::<Result<Vec<Vertex>>>()?;
        if !heur_queries.is_empty() {
            let heur_results = heur.estimate(target, &heur_queries)?;
            for (node, result) in heur_queries.iter().zip(heur_results.iter()) {
                debug_assert!(!heur_cache.contains_key(&node.id()));
                heur_cache.insert(node.id(), *result);
            }
            stats.heuristic_runs += 1;
            stats.heuristic_nodes += heur_queries.len();
        }


        // Finally, update the priority queue.
        for id in node_ids_to_relax {
            // Safety: We just inserted the heuristic value into the cache. We
            // also just updated the G score.
            let gscore = cur_gscore + 1;
            let fscore = gscore as f32 + heur_cache.get(&id).unwrap();
            debug_assert!(gscore == gs.get(&id).unwrap().gscore);
            pq.push(PQEntry { node_id: id, fscore });
        }
    }

    // We couldn't find a path
    Ok(AStarResult { path: None, stats })
}
