"""Memory dynamics shared by the long-term conversational agent."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DynamicsConfig:
    mode: str = "two-state"
    decay_rate: float = 0.12
    stability_protection: float = 0.85
    recovery_rate: float = 0.65
    stability_gain: float = 0.12
    top_k: int = 5

    def __post_init__(self):
        if self.mode not in ("weight-only", "two-state"):
            raise ValueError("mode must be weight-only or two-state")
        for name in ("decay_rate", "stability_protection", "recovery_rate", "stability_gain"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be nonnegative")
        if not 0 <= self.stability_protection < 1:
            raise ValueError("stability_protection must be in [0, 1)")
        if not 0 <= self.recovery_rate <= 1:
            raise ValueError("recovery_rate must be in [0, 1]")
        if not 0 <= self.stability_gain <= 1:
            raise ValueError("stability_gain must be in [0, 1]")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")

    def to_dict(self):
        return asdict(self)


def _state_value(memory, key):
    value = memory.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"memory {key} must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError(f"memory {key} must be in [0, 1]")
    return float(value)


def decay(memory, config):
    before = _state_value(memory, "weight")
    stability = _state_value(memory, "stability")
    effective_stability = stability if config.mode == "two-state" else 0.0
    rate = config.decay_rate * (1 - config.stability_protection * effective_stability)
    after = before * math.exp(-rate)
    memory["weight"] = after
    if config.mode == "weight-only":
        memory["stability"] = 0.0
    return {"weight_before": before, "weight_after": after, "effective_decay_rate": rate}


def activate(memory, config):
    before_weight = _state_value(memory, "weight")
    stored_stability = _state_value(memory, "stability")
    before_stability = stored_stability if config.mode == "two-state" else 0.0
    after_weight = before_weight + (1 - before_weight) * config.recovery_rate
    if config.mode == "two-state":
        gain = config.stability_gain * (1 - before_weight)
        after_stability = before_stability + (1 - before_stability) * gain
    else:
        after_stability = 0.0
    memory["weight"] = min(1.0, after_weight)
    memory["stability"] = min(1.0, after_stability)
    return {
        "weight_before": before_weight,
        "weight_after": memory["weight"],
        "stability_before": before_stability,
        "stability_after": memory["stability"],
        "stability_gain": memory["stability"] - before_stability,
    }


def _dot(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions changed; use a new database")
    return sum(a * b for a, b in zip(left, right))


def rank(memories, query, top_k=5):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    rows = []
    for memory in memories:
        weight = _state_value(memory, "weight")
        if query is None:
            relevance = 1.0
        else:
            similarity = _dot(query, memory["vector"])
            relevance = max(0.0, min(1.0, (1 + similarity) / 2))
        visible = {key: value for key, value in memory.items() if key != "vector"}
        rows.append({
            **visible,
            "relevance": relevance,
            "score": relevance * weight,
        })
    rows.sort(key=lambda row: (-row["score"], row["id"]))
    for index, row in enumerate(rows):
        row["selected"] = index < top_k
    return rows
