from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryEntry:
    key: str
    text: str
    tags: list[str] = field(default_factory=list)


class ConversationMemory:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, message: str) -> None:
        self.messages.append(message)

    def recent(self, limit: int = 3) -> list[str]:
        return self.messages[-limit:]


class LongTermMemory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def search(self, query: str) -> list[MemoryEntry]:
        query_terms = set(query.lower().split())
        ranked = sorted(
            self.entries,
            key=lambda entry: len(query_terms & set(entry.text.lower().split())) + len(query_terms & set(entry.tags)),
            reverse=True,
        )
        return [entry for entry in ranked if entry.text][:3]


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, state: dict[str, object]) -> None:
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self) -> dict[str, object]:
        return json.loads(self.path.read_text(encoding="utf-8"))


def main() -> None:
    conversation = ConversationMemory()
    conversation.add("User wants a deployment checklist for a RAG system.")
    conversation.add("Assistant suggested evals, tracing, and rollback planning.")
    conversation.add("User prefers low-cost hosting and wants citation support.")

    long_term = LongTermMemory()
    long_term.add(MemoryEntry("preference_1", "User prefers low-cost hosting options.", tags=["cost", "hosting"]))
    long_term.add(MemoryEntry("preference_2", "User values citations and traceability in responses.", tags=["citations", "trust"]))
    long_term.add(MemoryEntry("project_1", "Current project is a RAG assistant for support engineers.", tags=["rag", "support"]))

    checkpoint = CheckpointStore(Path(__file__).with_name("agent_state.json"))
    checkpoint.save({"recent_messages": conversation.recent(), "memory_hits": [entry.key for entry in long_term.search("rag citations")]})

    print("Recent conversation")
    for message in conversation.recent():
        print(f"- {message}")

    print("\nRelevant long-term memory")
    for entry in long_term.search("rag citations"):
        print(f"- {entry.key}: {entry.text}")

    print("\nCheckpoint")
    print(checkpoint.load())


if __name__ == "__main__":
    main()
