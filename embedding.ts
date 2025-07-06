import { AutoTokenizer, AutoModel } from '@xenova/transformers';
import type { Tensor } from '@xenova/transformers';

// Pool the hidden states of the last valid token for each sequence.
function lastTokenPool(lastHiddenStates: Tensor, attentionMask: Tensor): Tensor {
  const [batchSize, seqLen, hiddenSize] = lastHiddenStates.dims;
  const data = new Float32Array(batchSize * hiddenSize);

  // Determine if padding is added to the left. We check if every sequence has
  // attention on the last position.
  let lastColumnSum = 0;
  for (let i = 0; i < batchSize; ++i) {
    lastColumnSum += attentionMask.data[i * seqLen + seqLen - 1];
  }
  const leftPadding = lastColumnSum === batchSize;

  for (let i = 0; i < batchSize; ++i) {
    // Compute the index of the last valid token in the sequence.
    let index = seqLen - 1;
    if (!leftPadding) {
      let length = 0;
      for (let j = 0; j < seqLen; ++j) {
        length += attentionMask.data[i * seqLen + j];
      }
      index = length - 1;
    }
    const srcOffset = (i * seqLen + index) * hiddenSize;
    const dstOffset = i * hiddenSize;
    for (let j = 0; j < hiddenSize; ++j) {
      data[dstOffset + j] = lastHiddenStates.data[srcOffset + j];
    }
  }

  return new (lastHiddenStates.constructor as any)(lastHiddenStates.type, data, [batchSize, hiddenSize]);
}

function getDetailedInstruct(taskDescription: string, query: string): string {
  return `Instruct: ${taskDescription}\nQuery:${query}`;
}

async function main() {
  const task = 'Given a web search query, retrieve relevant passages that answer the query';

  const queries = [
    getDetailedInstruct(task, 'What is the capital of China?'),
    getDetailedInstruct(task, 'Explain gravity')
  ];

  const documents = [
    'The capital of China is Beijing.',
    'Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.'
  ];

  const inputTexts = queries.concat(documents);

  const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', { padding_side: 'left' });
  const model = await AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B');

  const batch = await tokenizer(inputTexts, { padding: true, truncation: true, max_length: 8192 });
  const outputs = await model(batch);
  let embeddings = lastTokenPool(outputs.last_hidden_state as Tensor, batch['attention_mask'] as Tensor);

  // Normalize embeddings
  const [batchSize, hiddenSize] = embeddings.dims;
  for (let i = 0; i < batchSize; ++i) {
    let norm = 0;
    for (let j = 0; j < hiddenSize; ++j) {
      const v = embeddings.data[i * hiddenSize + j];
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    for (let j = 0; j < hiddenSize; ++j) {
      embeddings.data[i * hiddenSize + j] /= norm;
    }
  }

  // Compute similarity scores between queries and documents
  const queryEmbeddings = embeddings.slice([0, 0], [2, hiddenSize]);
  const documentEmbeddings = embeddings.slice([2, 0], [embeddings.dims[0] - 2, hiddenSize]);
  const scores: number[][] = [];

  for (let i = 0; i < queryEmbeddings.dims[0]; ++i) {
    const row: number[] = [];
    for (let j = 0; j < documentEmbeddings.dims[0]; ++j) {
      let dot = 0;
      for (let k = 0; k < hiddenSize; ++k) {
        const q = queryEmbeddings.data[i * hiddenSize + k];
        const d = documentEmbeddings.data[j * hiddenSize + k];
        dot += q * d;
      }
      row.push(dot);
    }
    scores.push(row);
  }

  console.log(scores);
}

main().catch(err => {
  console.error(err);
});

