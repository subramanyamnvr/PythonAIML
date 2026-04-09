# Study Plan

This plan is aligned to the current repo layout, not the earlier placeholder naming scheme.

## Phase 0: Orientation

Duration:

- 1 day

Focus:

- read the root `README.md`
- read `00_repo_guide/repo_map.md`
- identify which folders are active learning paths and which are scaffolds

Deliverables:

- a personal learning order
- a short list of priority folders

## Phase 1: Python Core

Duration:

- 3 to 5 weeks

Primary folders:

- `01-python-deepdive-main/python-deepdive-main`
- `15_interview_prep`

Focus:

- syntax and control flow
- data structures
- functions and comprehensions
- scope, mutability, memory model
- OOP
- exceptions and modules
- decorators, iterators, generators

Deliverables:

- short notes for each section
- 3 to 5 small code examples per topic
- a first pass through Python interview questions

## Phase 2: Python Libraries

Duration:

- 2 to 4 weeks

Primary folders:

- `02-Numpy`
- `03-Pandas`
- `04-Matplotlib`
- `05-Seaborn`

Focus:

- arrays and vectorized thinking
- data wrangling
- plotting and EDA
- data storytelling basics

Deliverables:

- one cleaned dataset notebook
- one small EDA notebook
- quick NumPy and Pandas revision notes

## Phase 3: NLP + Classical ML

Duration:

- 4 to 6 weeks

Primary folders:

- `06-NLPComplete`
- `08_Machine_learning`
- `03_math_for_ai_ml`

Focus:

- text preprocessing
- bag-of-words, TF-IDF, Word2Vec
- regression and classification
- tree-based methods and boosting
- clustering, PCA, anomaly detection
- model evaluation and intuition

Deliverables:

- 2 model comparison notebooks
- 1 text classification walkthrough
- interview notes for metrics, bias/variance, and model tradeoffs

## Phase 4: Deep Learning Frameworks

Duration:

- 4 to 6 weeks

Primary folders:

- `07-LibrariesForDeepLearning`
- `09-ComputerVision`
- `10-deep_learning`

Focus:

- PyTorch workflow
- TensorFlow/Keras workflow
- transfer learning
- experiment tracking basics
- CNNs and image pipelines
- computer vision fundamentals
- sequence models, autoencoders, GANs, and transformers

Deliverables:

- 1 PyTorch notebook summary
- 1 TensorFlow notebook summary
- 1 image-focused mini-project or experiment log

## Phase 5: Typed Python + Pydantic + GenAI + Agentic

Duration:

- 3 to 5 weeks

Primary folders:

- `11_typed_python_and_pydantic`
- `12_genai`
- `13_Agentic`
- `Misc_resources`

Focus:

- Python type hints and modeling patterns
- dataclasses and related lightweight model libraries
- data validation and schema design
- settings and configuration models
- structured outputs for apps and LLM workflows
- LLM foundations
- prompt engineering
- embeddings and vector search
- RAG
- agents and tools
- LangGraph, CrewAI, AutoGen, and MCP
- evals, guardrails, and system tradeoffs

Deliverables:

- one Pydantic mini-project or validation workflow
- one RAG proof of concept
- one agent or workflow experiment
- GenAI interview and design notes

## Phase 6: MLOps + Projects

Duration:

- 2 to 4 weeks

Primary folders:

- `14-mlops_and_deployment`
- `16_projects`

Focus:

- packaging and Git workflow
- Docker and deployment basics
- serving and monitoring concepts
- work through `16_projects` in numbered order, starting with the strongest portfolio pieces
- turning notebooks into portfolio-ready work

Deliverables:

- one deployed or deployment-ready project
- one polished write-up from a top-ranked project in `16_projects`
- a short write-up for architecture, logging, and monitoring choices

## Phase 7: Interview Consolidation

Duration:

- ongoing

Primary folders:

- `15_interview_prep`

Focus:

- topic-wise revision
- question practice
- case studies
- weak-area review

Deliverables:

- concise answer bank
- topic comparison notes
- weekly weak-topic tracker

## Weekly Rhythm

Use a repeatable loop:

1. Learn from the numbered notebooks or notes.
2. Write short summaries in your own words.
3. Implement or rerun examples.
4. Extract interview questions.
5. Revise the weak parts.
6. Add one mini deliverable each week.

## Practical Note

For folders with a flat notebook layer, use the top-level numbered notebooks first. They were added to make the repo easier to browse without breaking the original local file dependencies.

For `10-deep_learning`, use the numbered topic folders first, then follow the notebooks inside each folder in project order.

For `11_typed_python_and_pydantic`, use the companion code-sample folders first when you want quick syntax examples, then move into the larger Pydantic modules.

For `12_genai`, use the numbered topic folders first, then move into the orchestration and platform sections once the retrieval path makes sense.

For `13_Agentic`, use the numbered week folders in order, and treat `guides` and `setup` as support folders rather than the main path.

For `16_projects`, use the numbered folders as a portfolio-first execution order, not as a difficulty ladder.
