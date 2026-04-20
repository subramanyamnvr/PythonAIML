# GenAI Path

This folder is now organized as one flat, numbered topic folder per module.

- The first layer is the main learning path.
- Loose notebooks, PDFs, and imported mini-projects were grouped into topic folders.
- Redundant wrapper folders were flattened where they only added one extra level without adding meaning.

Open [module_map.md](module_map.md) for the topic-by-topic breakdown, or open [learning_path.html](learning_path.html) for a visual view of the sequence.

## Recommended Order

### Stage 1: LLM and LangChain Foundations

- `01-LLM-Foundations`
- `02-LangChain-Basics-and-Ecosystem`
- `03-OpenAI-and-Ollama`

### Stage 2: Retrieval and Knowledge Workflows

- `04-RAG-Document-QA`
- `05-Conversational-RAG`
- `06-Search-and-Hybrid-Retrieval`
- `07-Structured-Data-and-Knowledge-Graphs`

### Stage 3: Output-Focused Applications

- `08-Summarization-and-Task-Apps`

### Stage 4: Models and Adaptation

- `09-Hugging-Face-and-Open-Source-Models`
- `10-Fine-Tuning`

### Stage 5: Orchestration and Platforms

- `11-LCEL-and-LangChain-Updates`
- `12-LangGraph-and-Orchestration-Notes`
- `13-Cloud-GenAI-Platforms`
- `14-Agentic-Framework-Extras`

### Stage 6: Advanced Production Topics

- `15-PEFT-LoRA-QLoRA-and-Inference-Serving`
- `16-LLM-Evals-and-Observability`
- `17-Guardrails-and-AI-Security`
- `18-Multimodal-GenAI`
- `19-Agent-Memory-and-State`

### Stage 7: Legacy Background Material

- `20-Legacy-Neural-Network-Background`

### Stage 8: Imported Course Packs

- `21-RAG-Mastery`
- `22-LangGraph-Tutorials`

## Practical Notes

- Start from the lowest-numbered folder unless you already know the basics.
- Within a topic folder, follow notebooks in `01`, `02`, `03` or `Project_01`, `Project_02`, `Project_03` order where that pattern exists.
- Some imported course material still uses nested dataset folders or local asset folders. Those were kept beside the code that depends on them.
- `15` through `19` are the newer production-oriented additions for adaptation, evals, safety, multimodal work, and stateful agent design.
- `20-Legacy-Neural-Network-Background` holds older ANN, RNN, and LSTM study material that was sitting in the GenAI area already. It was grouped cleanly, but it is not the main GenAI path.
- `21` and `22` are imported course packs that were appended without renumbering the existing curriculum, so the original material stays stable.
