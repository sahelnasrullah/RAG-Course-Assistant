# Retrieval results

45 answerable questions, 79 chunks, embedding model `intfloat/e5-base-v2`.

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| dense | 0.80 | 0.98 | 1.00 | 0.89 |
| bm25 | 0.84 | 0.98 | 1.00 | 0.91 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 |

Questions whose gold chunk is not in the top 4:

- dense: none
- bm25: none
- hybrid: none
