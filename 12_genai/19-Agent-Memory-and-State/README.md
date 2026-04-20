# Agent Memory And State

## Focus
Conversation state, memory strategies, and resumable execution patterns for agentic GenAI systems.

## Included Here
- `agent_memory_demo.py`: a runnable memory/state example with recent-message tracking, long-term lookup, and checkpoint persistence
- `README.md`: the quick-start guide for this topic folder

## Run
`python agent_memory_demo.py`

The demo writes its checkpoint to a temp-directory JSON file so it stays runnable without modifying the repo tree.

## Expand With
- retrieval-backed memory and summarization strategies
- checkpoints, resumability, and tool-state handling
- agent state schemas that carry metadata across turns
- longer-lived memory stores for user preferences and project context
