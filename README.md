# Shortest Paths with Text Embeddings on Simple English Wikipedia

This project aims to play Wikiracer using the text embeddings of the articles on
Simple English Wikipedia. It can train models for link prediction and distance
estimation, and it can run the distance estimation models on A*.

## Project Structure
  * `data`: Contains the downloaded raw data from The Wikimedia Foundation, as
    well as the MySQL and Memgraph database files. Also contains the
    intermediate data generated for training and evaluation.
  * `prep`: Contains scripts that were used while preparing the data for
    training, such as for retrieving the text embeddings from OpenAI or
    computing the PageRank of every node.
  * `eda`: Contains scripts for analyzing the distribution of distances and
    PageRanks on the graph, as well as scripts for characterizing the text
    embeddings.
  * `link-prediction`: Contains scripts to train and evaluate logistic
    regression models for link prediction.
  * `distance-estimation`: Same as above but for distance estimation.
  * `graph-search`: Contains code for running A* with the heuristics generated
    above or with the "Null" heuristic.
  * `docker-compose.yml`: Has profiles for Memgraph and for MySQL, pulling files
    from the `data` directory.
  * `config.yaml`: Has paths to all the data files.
