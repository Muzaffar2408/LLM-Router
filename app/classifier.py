from dataclasses import dataclass

from .models import QualityTier


@dataclass
class ClassificationResult:
    tier: QualityTier
    reason: str


class HeuristicClassifier:
    """
    Phase 2 fallback classifier.

    It is intentionally simple. Phase 3's traffic cop uses an LLM first and
    reaches this classifier only when the LLM classification call fails.
    """

    HIGH_SIGNALS = (
        "implement",
        "algorithm",
        "architecture",
        "distributed",
        "concurrent",
        "thread-safe",
        "proof",
        "derive",
        "optimize",
        "complex",
    )

    MEDIUM_SIGNALS = (
        "compare",
        "analyze",
        "explain",
        "summarize",
        "design",
        "debug",
        "calculate",
    )

    def predict(self, prompt: str) -> ClassificationResult:
        text = prompt.lower()

        if any(signal in text for signal in self.HIGH_SIGNALS):
            return ClassificationResult(
                tier=QualityTier.HIGH,
                reason="Matched high-complexity heuristic signals.",
            )

        if any(signal in text for signal in self.MEDIUM_SIGNALS):
            return ClassificationResult(
                tier=QualityTier.MEDIUM,
                reason="Matched medium-complexity heuristic signals.",
            )

        return ClassificationResult(
            tier=QualityTier.LOW,
            reason="No complex heuristic signals detected.",
        )
