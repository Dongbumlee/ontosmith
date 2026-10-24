"""Pre-registration as data structures.

The lab's measurement discipline ran on operator habit:
counted projections computed before any spend, stop-lines a
run may not cross, gates registered before a cell executes,
and a frontier-escalation rule that never lets the escalation
model sit in the judging seat. This module makes that
discipline product configuration — plain dataclasses that
serialize to JSON, validate fail-closed, and carry no corpus,
pack, or model names.

Layering: this is *product* configuration. The
domain layer (packs / profiles) declares capability tiers
only; prices and bindings are caller-supplied data here, and
secrets never appear (API keys are environment names, read by
the callers that need them).

Contents:

- :class:`GateDefinition` / :class:`GateResult` — one
  pre-registered adoption gate: a metric, a comparator, a
  threshold, and the consequence recorded when it fires.
- :class:`StopLine` — a spend cap plus the (lower) stop-line
  a run must halt at, with an optional reserve.
- :class:`CountedProjection` — a cost projection built from
  counted units (items, characters, tokens, calls) priced by
  caller-supplied unit costs; scenarios combine into a
  governing (maximum) figure.
- :class:`SpendLedger` — an append-only JSONL spend ledger
  with a stop-line guard and checkpoint/resume semantics.
- :class:`FrontierEscalationConfig` — optional, OFF by
  default; only a declared shaky band escalates, and the
  judging seat never escalates.
- :class:`VerificationConfig` — the aggregate a run records:
  gates + stop-line + escalation + seed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_COMPARATORS = (">=", ">", "<=", "<", "==")


def _compare(value: float, comparator: str, threshold: float) -> bool:
    """Apply one registered comparator; unknown ones raise."""
    if comparator == ">=":
        return value >= threshold
    if comparator == ">":
        return value > threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "==":
        return value == threshold
    raise ValueError(f"unregistered comparator {comparator!r}")


@dataclass(frozen=True)
class GateDefinition:
    """One pre-registered adoption gate.

    ``metric`` names the quantity a cell reports; the gate
    passes when ``metric comparator threshold`` holds.
    ``consequence`` records what happens when the gate decides
    (e.g. "NOT ADOPTED — the configuration stands"), so the
    ruling is data, not a comment in a design document.
    """

    name: str
    metric: str
    comparator: str
    threshold: float
    consequence: str = ""

    def __post_init__(self) -> None:
        if self.comparator not in _COMPARATORS:
            raise ValueError(
                f"gate {self.name!r}: comparator "
                f"{self.comparator!r} is not one of {_COMPARATORS}")
        if not self.name or not self.metric:
            raise ValueError("gate name and metric are required")

    def evaluate(self, metrics: dict) -> GateResult:
        """Evaluate this gate against a cell's reported metrics.

        A missing metric is a gate FAILURE with the defect
        named — never a silent pass (the fail-closed rule).
        """
        if self.metric not in metrics or metrics[self.metric] is None:
            return GateResult(
                gate=self.name, metric=self.metric,
                comparator=self.comparator, threshold=self.threshold,
                value=None, passed=False,
                note=f"metric {self.metric!r} not reported")
        value = float(metrics[self.metric])
        return GateResult(
            gate=self.name, metric=self.metric,
            comparator=self.comparator, threshold=self.threshold,
            value=value, passed=_compare(value, self.comparator,
                                         self.threshold),
            note="")


@dataclass(frozen=True)
class GateResult:
    """The recorded outcome of one gate evaluation."""

    gate: str
    metric: str
    comparator: str
    threshold: float
    value: float | None
    passed: bool
    note: str = ""


def evaluate_gates(gates: list[GateDefinition],
                   metrics: dict) -> list[GateResult]:
    """Evaluate every registered gate, in registration order."""
    return [g.evaluate(metrics) for g in gates]


def all_passed(results: list[GateResult]) -> bool:
    """True iff every evaluated gate passed (adoption rule)."""
    return bool(results) and all(r.passed for r in results)


@dataclass(frozen=True)
class StopLine:
    """A spend cap and the stop-line a run must halt at.

    The cap is the hard budget; the stop-line (default = cap)
    is where execution halts, leaving room for in-flight work.
    ``reserve_usd`` is held back from every start decision: a
    new unit of work starts only if ``spent + estimate +
    reserve <= stop_line``.
    """

    cap_usd: float
    stop_line_usd: float | None = None
    reserve_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.cap_usd < 0 or self.reserve_usd < 0:
            raise ValueError("cap and reserve must be >= 0")
        if self.stop_line > self.cap_usd:
            raise ValueError(
                f"stop_line ${self.stop_line:.4f} exceeds cap "
                f"${self.cap_usd:.4f}")

    @property
    def stop_line(self) -> float:
        """The effective stop-line (cap when undeclared)."""
        return self.cap_usd if self.stop_line_usd is None \
            else self.stop_line_usd

    def may_start(self, spent_usd: float, estimate_usd: float) -> bool:
        """True iff starting work estimated at ``estimate_usd``
        keeps the run at or under the stop-line."""
        return spent_usd + estimate_usd + self.reserve_usd \
            <= self.stop_line

    def remaining(self, spent_usd: float) -> float:
        """Budget left before the stop-line, reserve excluded."""
        return round(self.stop_line - self.reserve_usd - spent_usd, 6)


class StopLineExceeded(RuntimeError):
    """Raised when a ledger append would cross the stop-line."""


@dataclass
class CountedProjection:
    """A counted cost projection: units counted from the
    real builders, priced by caller-supplied unit costs.

    ``scenarios`` maps a scenario name to its line items
    ``{unit: count}``; ``unit_costs`` maps a unit to its USD
    price. The governing figure is the maximum scenario total
    — the governing projection is max(A,
    B'), and a cell whose governing projection exceeds its cap
    does not start.
    """

    unit_costs: dict[str, float]
    scenarios: dict[str, dict[str, float]] = field(default_factory=dict)

    def scenario_total(self, name: str) -> float:
        """USD total of one scenario; unknown units fail closed."""
        items = self.scenarios[name]
        unknown = sorted(set(items) - set(self.unit_costs))
        if unknown:
            raise ValueError(
                f"projection scenario {name!r} counts units with "
                f"no price: {unknown} — a counted projection may "
                f"not silently price work at $0")
        return round(sum(items[u] * self.unit_costs[u]
                         for u in items), 6)

    def totals(self) -> dict[str, float]:
        """Every scenario's USD total, by scenario name."""
        return {name: self.scenario_total(name)
                for name in self.scenarios}

    def governing(self) -> float:
        """The governing (maximum) scenario total; 0 when empty."""
        totals = self.totals()
        return max(totals.values()) if totals else 0.0

    def assert_within(self, stop: StopLine) -> float:
        """Return the governing total, or raise when it exceeds
        the stop-line — the prepare-stage STOP of the lab cells."""
        governing = self.governing()
        if governing > stop.stop_line:
            raise StopLineExceeded(
                f"counted projection ${governing:.4f} exceeds "
                f"stop-line ${stop.stop_line:.4f}; the cell does "
                f"not start")
        return governing


class SpendLedger:
    """Append-only JSONL spend ledger with stop-line guard.

    Extracted from the per-cell ledgers: one
    line per unit of work, flushed per append, carrying a
    cumulative total rounded to 6 decimals; re-opening an
    existing ledger resumes its total (checkpoint/resume), so
    a killed run's spend is never double-counted or lost.
    """

    def __init__(self, path: str | Path | None,
                 stop: StopLine | None = None):
        self.path = Path(path) if path is not None else None
        self.stop = stop
        self._total = self._seed_total()

    def _seed_total(self) -> float:
        if self.path is None or not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text().splitlines():
            if line.strip():
                total = float(json.loads(line)["cumulative_usd"])
        return total

    def total(self) -> float:
        """The ledger's cumulative spend in USD."""
        return self._total

    def completed_ids(self) -> set[str]:
        """Item ids already recorded with status ``ok`` — the
        resume set: a relaunched cell skips exactly these."""
        done: set[str] = set()
        if self.path is None or not self.path.exists():
            return done
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("status", "ok") == "ok" \
                    and entry.get("item_id") is not None:
                done.add(str(entry["item_id"]))
        return done

    def append(self, entry: dict) -> float:
        """Append one entry, enforcing the stop-line first.

        The entry's ``cost_usd`` is added to the cumulative
        total (recorded on the line as ``cumulative_usd``). An
        append that would cross the stop-line raises
        :class:`StopLineExceeded` and writes nothing.
        """
        cost = float(entry.get("cost_usd", 0.0))
        new_total = round(self._total + cost, 6)
        if self.stop is not None and new_total > self.stop.stop_line:
            raise StopLineExceeded(
                f"ledger append of ${cost:.4f} would take the "
                f"total to ${new_total:.4f}, over the stop-line "
                f"${self.stop.stop_line:.4f}")
        entry = {**entry, "cost_usd": cost,
                 "cumulative_usd": new_total}
        self._total = new_total
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
        return new_total


#: Roles that sit in the judging seat. Escalation to a frontier
#: model is drafting/advising only — these roles NEVER escalate
#: (doctrine: the judging seat itself never
#: escalates, whatever the config says).
JUDGING_ROLES = frozenset({
    "judge", "pair_judge", "gate", "jev", "er_gate",
    "triple_gate", "conflict_gate", "onto_gate",
})


@dataclass(frozen=True)
class FrontierEscalationConfig:
    """Frontier-escalation policy.

    Optional and **off by default**. When enabled, only items
    whose first-pass score lands in the shaky band
    ``[band_lo, band_hi]`` are escalated, and only for
    non-judging roles (audit adjudication, drafting,
    advising). The promotion rule caps tolerated flips: if
    the escalation verdicts overturn more than
    ``max_overturn_rate`` of a seeded sample of uncontested
    verdicts, all remaining verdicts escalate (the P2 5%
    rule, as data).
    """

    enabled: bool = False
    band_lo: float = 0.25
    band_hi: float = 0.50
    sample_rate: float = 0.05
    max_overturn_rate: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.band_lo <= self.band_hi <= 1.0:
            raise ValueError(
                f"escalation band [{self.band_lo}, {self.band_hi}] "
                f"is not a valid sub-range of [0, 1]")
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        if not 0.0 <= self.max_overturn_rate <= 1.0:
            raise ValueError("max_overturn_rate must be in [0, 1]")

    def should_escalate(self, role: str, score: float | None) -> bool:
        """True iff this item escalates under the policy.

        Judging roles never escalate; a missing score never
        escalates (an unscored item is a failure to record,
        not a shaky verdict).
        """
        if not self.enabled or score is None:
            return False
        if role in JUDGING_ROLES:
            return False
        return self.band_lo <= score <= self.band_hi


@dataclass
class VerificationConfig:
    """The aggregate pre-registration a verification run records.

    Gates and the stop-line are the cell's contract, fixed
    before execution; the escalation policy rides along
    (default off). Serializes to plain JSON for run records.
    """

    gates: list[GateDefinition] = field(default_factory=list)
    stop_line: StopLine | None = None
    escalation: FrontierEscalationConfig = field(
        default_factory=FrontierEscalationConfig)
    seed: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain JSON-able dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VerificationConfig:
        """Rebuild from a plain dict; unknown gates fields and
        invalid values fail closed via the dataclasses."""
        return cls(
            gates=[GateDefinition(**g) for g in d.get("gates", [])],
            stop_line=StopLine(**d["stop_line"])
            if d.get("stop_line") else None,
            escalation=FrontierEscalationConfig(
                **d.get("escalation", {})),
            seed=int(d.get("seed", 0)))
