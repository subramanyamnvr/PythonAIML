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

- `11-LCEL-and-LangChain-Updates`  
  LCEL, serve/client experiments, and updated LangChain project structure.

## Orchestration and Platforms

- `12-LangGraph-and-Orchestration-Notes`  
  LangGraph starter notebooks and orchestration-oriented workflow examples.

- `13-Cloud-GenAI-Platforms`  
  AWS Bedrock, Nvidia NIM, and broader cloud GenAI project material.

- `14-Agentic-Framework-Extras`  
  CrewAI crash-course material and MCP notes that were already colocated in the GenAI area.

## Legacy Background Material

- `15-Legacy-Neural-Network-Background`  
  Older ANN, RNN, LSTM, and sequence-model reference material that did not belong in the main GenAI path but was worth preserving in one clear place.

## Layout Notes

- The first folder layer is now continuously numbered from `01` to `15`.
- Related loose notebooks and PDFs were moved into topic folders so the root stays readable.
- Wrapper folders were flattened where they only repeated the folder name without adding structure.
- Imported mini-project repos remain intact inside the topic folders when they contain their own local assets or app structure.
