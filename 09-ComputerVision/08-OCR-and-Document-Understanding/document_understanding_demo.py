from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRToken:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


def reading_order(tokens: list[OCRToken], line_tolerance: int = 12) -> list[list[OCRToken]]:
    rows: list[list[OCRToken]] = []
    for token in sorted(tokens, key=lambda item: (item.y1, item.x1)):
        for row in rows:
            if abs(row[0].center_y - token.center_y) <= line_tolerance:
                row.append(token)
                row.sort(key=lambda item: item.x1)
                break
        else:
            rows.append([token])
    return rows


def extract_key_values(rows: list[list[OCRToken]]) -> dict[str, str]:
    results: dict[str, str] = {}
    for row in rows:
        texts = [token.text for token in row]
        if ":" in "".join(texts):
            joined = " ".join(texts).replace(" :", ":")
            key, value = joined.split(":", 1)
            results[key.strip()] = value.strip()
        elif len(row) >= 2:
            key = row[0].text.rstrip(":")
            value = " ".join(token.text for token in row[1:])
            if key.lower() in {"invoice", "vendor", "total", "date"}:
                results[key] = value
    return results


def sample_tokens() -> list[OCRToken]:
    return [
        OCRToken("Invoice", 10, 10, 70, 28),
        OCRToken("INV-1007", 90, 10, 160, 28),
        OCRToken("Vendor:", 10, 40, 70, 58),
        OCRToken("Northwind", 90, 40, 170, 58),
        OCRToken("Date:", 10, 70, 50, 88),
        OCRToken("2026-04-12", 90, 70, 180, 88),
        OCRToken("Total:", 10, 100, 55, 118),
        OCRToken("$1,248.40", 90, 100, 160, 118),
    ]


def main() -> None:
    rows = reading_order(sample_tokens())
    print("Reading order")
    for row in rows:
        print("-", " ".join(token.text for token in row))

    print("\nExtracted fields")
    for key, value in extract_key_values(rows).items():
        print(f"- {key:8s} -> {value}")


if __name__ == "__main__":
    main()
