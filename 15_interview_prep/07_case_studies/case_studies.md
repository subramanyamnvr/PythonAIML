# Case Studies

## Case 1: CTR Dropped After Deployment

- Confirm the drop is real and not a logging bug.
- Compare feature distributions and threshold behavior.
- Check whether serving-time preprocessing still matches training-time preprocessing.

## Case 2: RAG Answers Became Less Grounded

- Inspect retriever relevance first.
- Compare chunking and metadata filters.
- Review whether the prompt is forcing evidence citation.

## Case 3: GPU Cost Doubled

- Check batching efficiency, sequence lengths, caching, and concurrency.
- Measure whether the newest prompt or model choice inflated tokens.
