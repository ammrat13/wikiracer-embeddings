# Text-Embed Scripts

These scripts were used to push the entirety of Simple-English Wikipedia through
OpenAI's text embedding service. Articles were truncated to the maximum of 8191
tokens, then embedded using the Batch API.

The resulting vectors were saved as a `numpy` array on disk. The long
embeddings, consisting of all the components, were also saved in MemGraph.
