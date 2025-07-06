# tinetune-llm

This repository contains minimal examples for working with language models and embeddings. The `embedding.ts` script demonstrates how to load the Qwen3-Embedding-0.6B model using [Transformers.js](https://github.com/xenova/transformers.js).

## Requirements

- Node.js 18 or newer (tested with Node 22)
- npm (comes with Node) or another package manager
- ts-node (installed as a development dependency)

The script relies on the `@xenova/transformers` package which provides CPU and Apple Silicon GPU (Metal) backends. When running on an Apple M series machine the library will automatically use the Metal backend if available.

## Installation

1. Install dependencies:
   ```bash
   npm install
   ```
   The first run may take a while because the model weights are downloaded.

2. Run the script:
   ```bash
   npm start
   ```
   or
   ```bash
   npx ts-node embedding.ts
   ```

This will download the `Qwen/Qwen3-Embedding-0.6B` model and compute similarity scores between example queries and documents.

### Notes

If you see errors about missing packages make sure `npm install` completed successfully. On first execution the model weights are cached in `~/.cache/huggingface`.
When running on Apple M series hardware the backend automatically uses Metal for best performance.
