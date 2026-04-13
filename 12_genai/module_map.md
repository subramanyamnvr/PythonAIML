# GenAI Module Map

This map explains the cleaned GenAI sequence and what each numbered folder is for.

## Foundations

- `01-LLM-Foundations`  
  Intro PDFs and overview material for LLM evolution, transformer-era context, and the high-level roadmap.

- `02-LangChain-Basics-and-Ecosystem`  
  Core LangChain basics, ingestion, transformation, embeddings, vector stores, and the surrounding ecosystem notes.

- `03-OpenAI-and-Ollama`  
  OpenAI and Ollama starter notebooks, plus chatbot-oriented mini-apps.

## Retrieval and Knowledge Workflows

- `04-RAG-Document-QA`  
  Document-QA workflows, PDF querying, vector retrieval, and research-paper examples.

- `05-Conversational-RAG`  
  Retrieval with conversational memory and follow-up question flow.

- `06-Search-and-Hybrid-Retrieval`  
  Search engines, hybrid retrieval, and related notes on retrieval quality.

- `07-Structured-Data-and-Knowledge-Graphs`  
  Chat-with-SQL, graph-oriented notebooks, and Neo4j-style supporting files.

## Output-Focused Applications

- `08-Summarization-and-Task-Apps`  
  Summarization and task-specific GenAI examples such as MathsGPT.

## Models and Adaptation

- `09-Hugging-Face-and-Open-Source-Models`  
  Hugging Face with LangChain, CodeLlama material, and open-model efficiency notes such as quantization.

- `10-Fine-Tuning`  
  Fine-tuning code plus the Gemma notebook and related setup files.

## Orchestration and Platforms

- `11-LCEL-and-LangChain-Updates`  
  LCEL, serve/client experiments, and updated LangChain project structure.

- `12-LangGraph-and-Orchestration-Notes`  
  LangGraph starter notebooks and orchestration-oriented workflow examples.

- `13-Cloud-GenAI-Platforms`  
  AWS Bedrock, Nvidia NIM, and broader cloud GenAI project material.

- `14-Agentic-Framework-Extras`  
  CrewAI crash-course material and MCP notes that were already colocated in the GenAI area.

## Advanced Production Topics

- `15-PEFT-LoRA-QLoRA-and-Inference-Serving`  
  Starter expansion folder for PEFT, adapters, low-rank fine-tuning, quantized adaptation, and inference-serving patterns.

- `16-LLM-Evals-and-Observability`  
  Starter expansion folder for offline evals, online evals, tracing, regression checks, and application-level observability.

- `17-Guardrails-and-AI-Security`  
  Starter expansion folder for prompt-injection defenses, output validation, PII handling, policy checks, and safe tool use.

- `18-Multimodal-GenAI`  
  Starter expansion folder for image-text workflows, vision-language prompting, audio pipelines, and multimodal app patterns.

- `19-Agent-Memory-and-State`  
  Starter expansion folder for short-term memory, long-term memory, episodic traces, state stores, and checkpointed agent workflows.

## Legacy Background Material

- `20-Legacy-Neural-Network-Background`  
  Older ANN, RNN, LSTM, and sequence-model reference material that did not belong in the main GenAI path but was worth preserving in one clear place.

## Layout Notes

- The first folder layer is now continuously numbered from `01` to `20`.
- Related loose notebooks and PDFs were moved into topic folders so the root stays readable.
- Wrapper folders were flattened where they only repeated the folder name without adding structure.
- Imported mini-project repos remain intact inside the topic folders when they contain their own local assets or app structure.
- The newer `15` through `19` folders are starter expansion areas for modern production-grade GenAI work.
