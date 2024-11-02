# Text-Embed Scripts

These scripts were used to push the entirety of Simple-English Wikipedia through
OpenAI's text embedding service. Articles were truncated to the maximum of 8191
tokens, then embedded using the Batch API.

The resulting vectors were not put in MemGraph. Instead, they are saved as a
`numpy` array on disk. This is to save memory.
