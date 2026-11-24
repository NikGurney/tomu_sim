"""Readable inference transcript for the generate-and-filter loop (Definition 4, manuscript §3.4)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TraceEntry:
    """One evaluated candidate and its resulting confidence."""

    ell: int
    candidate: str
    fit: float
    confidence: float


@dataclass
class InferenceTrace:
    """Collect candidate evaluations and the final halt reason."""

    entries: List[TraceEntry] = field(default_factory=list)
    halt_reason: Optional[str] = None

    def log(self, ell: int, candidate: str, fit: float, confidence: float) -> None:
        """Append one candidate evaluation."""

        self.entries.append(TraceEntry(ell, candidate, fit, confidence))

    def halt(self, reason: str) -> None:
        """Record why the inference loop stopped."""

        self.halt_reason = reason

    def transcript(self) -> str:
        """Render the trace as stable, human-readable text."""

        lines = ["ell  candidate   fit       confidence"]
        for entry in self.entries:
            lines.append(
                f"{entry.ell:>3}  {entry.candidate:<10}  "
                f"{entry.fit:>8.6f}  {entry.confidence:>10.6f}"
            )
        lines.append(f"halt: {self.halt_reason or 'not halted'}")
        return "\n".join(lines)
