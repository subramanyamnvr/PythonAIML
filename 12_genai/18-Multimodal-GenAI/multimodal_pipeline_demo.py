from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    description: str
    width: int
    height: int


@dataclass
class AudioPart:
    transcript_hint: str
    duration_seconds: float


def compose_prompt(parts: list[object]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            chunks.append(f"TEXT: {part.text}")
        elif isinstance(part, ImagePart):
            chunks.append(f"IMAGE: {part.description} ({part.width}x{part.height})")
        elif isinstance(part, AudioPart):
            chunks.append(f"AUDIO: hint={part.transcript_hint} duration={part.duration_seconds}s")
    return "\n".join(chunks)


def route_task(parts: list[object]) -> str:
    has_image = any(isinstance(part, ImagePart) for part in parts)
    has_audio = any(isinstance(part, AudioPart) for part in parts)
    if has_image and has_audio:
        return "multimodal_reasoning"
    if has_image:
        return "vision_language"
    if has_audio:
        return "speech_pipeline"
    return "text_only"


def main() -> None:
    parts = [
        TextPart("Summarize the customer issue and identify the product shown."),
        ImagePart("photo of a damaged router with blinking red lights", 1024, 768),
        AudioPart("customer says the internet drops every hour", 18.5),
    ]
    print("Selected route:", route_task(parts))
    print("\nPrompt payload")
    print(compose_prompt(parts))


if __name__ == "__main__":
    main()
