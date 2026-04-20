from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MemoryEntry:
    key: str
    text: str
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    session_id: str
    recent_messages: list[str] = field(default_factory=list)
    memory_hits: list[str] = field(default_factory=list)
    summary: str = ""


class ConversationMemory:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, role: str, message: str) -> None:
        self.messages.append(f"{role}: {message}")

    def recent(self, limit: int = 3) -> list[str]:
        return self.messages[-limit:]

    def summarize(self, limit: int = 3) -> str:
        recent = self.recent(limit)
        return " | ".join(recent) if recent else "No recent messages."


class LongTermMemory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def search(self, query: str, limit: int = 3) -> list[MemoryEntry]:
        query_terms = set(query.lower().split())

        def score(entry: MemoryEntry) -> tuple[int, int]:
            text_terms = set(entry.text.lower().split())
            tag_terms = set(tag.lower() for tag in entry.tags)
            overlap = len(query_terms & text_terms) + len(query_terms & tag_terms)
            return overlap, len(entry.text)

        ranked = sorted(self.entries, key=score, reverse=True)
        return [entry for entry in ranked if score(entry)[0] > 0][:limit]


class CheckpointStore:
    def __init__(self, path: Path | None = None) -> None:
        default_path = Path(tempfile.gettempdir()) / "genai-agent-memory" / "agent_state.json"
        self.path = path or default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState) -> None:
        payload = asdict(state)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> AgentState:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return AgentState(**payload)

    def exists(self) -> bool:
        return self.path.exists()


def build_demo_state() -> tuple[ConversationMemory, LongTermMemory, AgentState]:
    conversation = ConversationMemory()
    conversation.add("user", "I need a deployment checklist for a RAG system.")
    conversation.add("assistant", "We should include evals, tracing, rollback, and a citation policy.")
    conversation.add("user", "I prefer low-cost hosting and strong source support.")
    conversation.add("assistant", "I will keep the plan lightweight and keep citations visible.")

    long_term = LongTermMemory()
    long_term.add(MemoryEntry("preference_1", "User prefers low-cost hosting options.", tags=["cost", "hosting"]))
    long_term.add(MemoryEntry("preference_2", "User values citations and traceability in responses.", tags=["citations", "trust"]))
    long_term.add(MemoryEntry("project_1", "Current project is a RAG assistant for support engineers.", tags=["rag", "support"]))
    long_term.add(MemoryEntry("preference_3", "User likes concise but practical step-by-step answers.", tags=["style", "format"]))

    query = "rag citations hosting"
    hits = long_term.search(query)
    state = AgentState(
        session_id="demo-session",
        recent_messages=conversation.recent(),
        memory_hits=[f"{entry.key}: {entry.text}" for entry in hits],
        summary=conversation.summarize(),
    )
    return conversation, long_term, state


def main() -> None:
    conversation, long_term, state = build_demo_state()
    checkpoint = CheckpointStore()
    checkpoint.save(state)

    print("Recent conversation")
    for message in conversation.recent():
        print(f"- {message}")

    print("\nRelevant long-term memory")
    for entry in long_term.search("rag citations hosting"):
        print(f"- {entry.key}: {entry.text}")

    print("\nConversation summary")
    print(state.summary)

    print("\nCheckpoint")
    print(json.dumps(asdict(checkpoint.load()), indent=2))


if __name__ == "__main__":
    main()
