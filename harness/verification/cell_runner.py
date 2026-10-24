"""The pre-registered adoption-cell runner.

The lab ran its configuration decisions as *cells*: a design
registered in advance (hypothesis, counted projection, gates,
cap), an execution that stops at a hard line, and a verdict
that reads the registered gates against the measured metrics
— adopted or not, by the numbers, never by re-argument. This
module is that discipline as a reusable core:

- a :class:`CellSpec` carries the registration (gates,
  stop-line, projection) as data;
- :func:`prepare_cell` refuses to start a cell whose counted
  projection exceeds its stop-line (the lab prepare-stage
  STOP);
- :func:`run_cell` executes items through a caller-supplied
  executor, appending each unit's cost to a
  :class:`~harness.verification.config.SpendLedger` and
  refusing to START a unit that would cross the stop-line
  (the lab per-document stop check); already-ledgered items
  are skipped, so a killed cell resumes from its ledger;
- :func:`judge_cell` evaluates the registered gates against
  the run's metrics and returns the verdict record.

The executor is the caller's: it receives one item and
returns ``(result, cost_usd)``. Domain work (what an item IS)
never enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import CountedProjection, GateDefinition, StopLine, StopLineExceeded, evaluate_gates


@dataclass
class CellSpec:
    """One cell's pre-registration.

    ``default_item_cost_usd`` prices an item when no per-item
    estimate is available — it is the stop-check estimate, and
    must be a counted figure (from the projection), not a
    guess: registration is the place where honesty is cheap.
    """

    name: str
    gates: list[GateDefinition] = field(default_factory=list)
    stop_line: StopLine = field(
        default_factory=lambda: StopLine(cap_usd=0.0))
    projection: CountedProjection | None = None
    default_item_cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a cell requires a name")
        if self.default_item_cost_usd < 0:
            raise ValueError("default_item_cost_usd must be >= 0")


def prepare_cell(spec: CellSpec) -> dict:
    """The prepare stage: verify the counted projection fits.

    Returns the projection record (per-scenario totals and
    the governing figure). Raises
    :class:`~harness.verification.config.StopLineExceeded`
    when the governing projection exceeds the stop-line —
    the cell does not start, and no spend has occurred.
    """
    record = {"cell": spec.name, "started": False}
    if spec.projection is not None:
        governing = spec.projection.assert_within(spec.stop_line)
        record["projection"] = {
            "scenarios": spec.projection.totals(),
            "governing_usd": governing,
        }
    record["stop_line_usd"] = spec.stop_line.stop_line
    record["cap_usd"] = spec.stop_line.cap_usd
    record["started"] = True
    return record


def run_cell(spec: CellSpec, items: list[dict], execute,
             ledger, item_id: str = "item_id",
             estimate=None) -> dict:
    """Execute a cell's items under its stop-line and ledger.

    ``execute(item) -> (result, cost_usd)``. ``estimate(item)``
    optionally prices the next item for the stop check
    (default: the spec's ``default_item_cost_usd``). Items
    already in the ledger with status ``ok`` are skipped
    (resume). The run stops — without executing — at the
    first item whose start would cross the stop-line; the
    returned record says exactly where and why it stopped.
    A second guard sits in the ledger itself: if an item's
    ACTUAL cost crosses the stop-line despite its estimate,
    the append raises StopLineExceeded and the run fails
    loud rather than recording spend off-book.
    Executor exceptions are recorded as ledger errors with
    the cost the executor reports via the exception's
    ``cost_usd`` attribute (default 0) and re-raised only
    when ``spec.metadata['fail_fast']`` is set.
    """
    done = ledger.completed_ids()
    results: dict[str, object] = {}
    executed = skipped = 0
    stopped = None
    for item in items:
        iid = str(item[item_id])
        if iid in done:
            skipped += 1
            continue
        est = estimate(item) if estimate else spec.default_item_cost_usd
        if not spec.stop_line.may_start(ledger.total(), est):
            stopped = {"item_id": iid, "reason": "stop_line",
                       "spent_usd": ledger.total(),
                       "estimate_usd": est,
                       "stop_line_usd": spec.stop_line.stop_line}
            break
        try:
            result, cost = execute(item)
        except Exception as exc:  # recorded, then policy decides
            ledger.append({"item_id": iid, "status": "error",
                           "error": f"{type(exc).__name__}: {exc}",
                           "cost_usd": float(
                               getattr(exc, "cost_usd", 0.0))})
            if spec.metadata.get("fail_fast"):
                raise
            continue
        ledger.append({"item_id": iid, "status": "ok",
                       "cost_usd": float(cost)})
        results[iid] = result
        executed += 1
    return {"cell": spec.name, "executed": executed,
            "skipped": skipped, "results": results,
            "spent_usd": ledger.total(), "stopped": stopped}


def judge_cell(spec: CellSpec, metrics: dict) -> dict:
    """The verdict stage: registered gates vs measured metrics.

    Adoption requires every gate to pass; the record carries
    each gate's outcome so the ruling is auditable without
    re-running anything.
    """
    results = evaluate_gates(spec.gates, metrics)
    return {
        "cell": spec.name,
        "adopted": bool(results) and all(r.passed for r in results),
        "gates": [{"gate": r.gate, "metric": r.metric,
                   "comparator": r.comparator,
                   "threshold": r.threshold, "value": r.value,
                   "passed": r.passed, "note": r.note}
                  for r in results],
        "metrics": metrics,
    }


def run_registered_cell(spec: CellSpec, items: list[dict], execute,
                        ledger_path: str | Path, metrics_of,
                        item_id: str = "item_id",
                        estimate=None) -> dict:
    """Full cell lifecycle: prepare -> run -> judge.

    ``metrics_of(run_record) -> metrics dict`` lets the caller
    compute the gated quantities from the run's own outputs.
    The ledger is bound to the spec's stop-line.
    """
    from .config import SpendLedger
    prep = prepare_cell(spec)
    ledger = SpendLedger(ledger_path, stop=spec.stop_line)
    run = run_cell(spec, items, execute, ledger,
                   item_id=item_id, estimate=estimate)
    verdict = judge_cell(spec, metrics_of(run))
    return {"prepare": prep, "run": run, "verdict": verdict}


__all__ = ["CellSpec", "StopLineExceeded", "judge_cell",
           "prepare_cell", "run_cell", "run_registered_cell"]
