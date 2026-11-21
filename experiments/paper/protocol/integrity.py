"""Attempt journals, append-only execution receipts, and compact evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any, Callable, Mapping
from uuid import uuid4
from hashlib import sha256

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, write_csv, write_pretty_json

from .validation import PaperContractError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_sidecar(path: Path) -> None:
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


@dataclass
class AttemptJournal:
    path: Path
    campaign_identity: str
    target_identity: str
    method: str
    expected_inference_calls: int
    started_at: str = field(default_factory=utc_now)
    started_clock: float = field(default_factory=perf_counter)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = {
            "schema_version": "mutaco-paper-attempt-journal-v1",
            "campaign_identity": self.campaign_identity,
            "target_record_identity": self.target_identity,
            "method": self.method,
            "status": "running",
            "phase": "method_dispatch",
            "started_at": self.started_at,
            "updated_at": self.started_at,
            "elapsed_seconds": 0.0,
            "last_completed_operation": None,
            "accounting_completeness": "partial_known_lower_bound",
            "accounting": {
                "inference_calls_started": 0,
                "inference_calls_completed": 0,
                "gp_fits_started": 0,
                "gp_fits_completed": 0,
                "proposals_started": 0,
                "proposals_completed": 0,
                "recommendations_started": 0,
                "recommendations_completed": 0,
                "terminal_recommendations_started": 0,
                "terminal_recommendations_completed": 0,
                "assessment_calls_started": 0,
                "assessment_calls_completed": 0,
            },
            "expected": {"inference_calls": self.expected_inference_calls},
            "last_event": None,
            "failure": None,
        }
        self.persist()

    def persist(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["elapsed_seconds"] = perf_counter() - self.started_clock
        write_pretty_json(self.path, self.state)

    def update(self, event: Mapping[str, Any]) -> None:
        accounting = self.state["accounting"]
        for key in accounting:
            if key in event:
                accounting[key] = int(event[key])
        if event.get("phase") is not None:
            self.state["phase"] = str(event["phase"])
        runner_event = event.get("runner_event")
        if isinstance(runner_event, Mapping):
            kind = str(runner_event.get("kind"))
            self.state["last_event"] = dict(runner_event)
            pairs = {
                "gp_fit_started": ("gp_fits_started", None),
                "gp_fit_finished": ("gp_fits_completed", "gp_fit"),
                "proposal_started": ("proposals_started", None),
                "proposal_finished": ("proposals_completed", "proposal"),
                "recommendation_started": ("recommendations_started", None),
            }
            if kind in pairs:
                counter, completed = pairs[kind]
                accounting[counter] += 1
                if completed is not None:
                    self.state["last_completed_operation"] = completed
            if kind == "simulator_attempt_finished":
                self.state["last_completed_operation"] = "simulator_evaluation"
            if kind == "run_finished" and accounting["recommendations_started"]:
                accounting["recommendations_completed"] = accounting[
                    "recommendations_started"
                ]
                self.state["last_completed_operation"] = "core_recommendation"
        self.persist()

    def finalize(
        self,
        *,
        status: str,
        raw: Mapping[str, Any] | None,
        exception: BaseException | None = None,
        opaque_failure: bool = False,
    ) -> dict[str, Any]:
        accounting = self.state["accounting"]
        if raw is not None:
            inference = int(raw.get("inference_calls", 0))
            assessment = int(raw.get("assessment_calls", 0))
            gp_fits = int(raw.get("gp_fits", 0))
            for prefix, value in (
                ("inference_calls", inference),
                ("assessment_calls", assessment),
                ("gp_fits", gp_fits),
            ):
                accounting[f"{prefix}_started"] = max(
                    accounting[f"{prefix}_started"], value
                )
                accounting[f"{prefix}_completed"] = max(
                    accounting[f"{prefix}_completed"], value
                )
        self.state["status"] = status
        unfinished = any(
            accounting[started] != accounting[completed]
            for started, completed in (
                ("inference_calls_started", "inference_calls_completed"),
                ("gp_fits_started", "gp_fits_completed"),
                ("proposals_started", "proposals_completed"),
                ("recommendations_started", "recommendations_completed"),
                (
                    "terminal_recommendations_started",
                    "terminal_recommendations_completed",
                ),
                ("assessment_calls_started", "assessment_calls_completed"),
            )
        )
        reported_completeness = (
            raw.get("accounting_completeness") if raw is not None else None
        )
        if reported_completeness not in {
            None,
            "exact",
            "partial_known_lower_bound",
            "partial_unknown",
        }:
            raise PaperContractError(
                f"unknown accounting completeness: {reported_completeness!r}"
            )
        self.state["accounting_completeness"] = (
            "partial_unknown"
            if opaque_failure
            else str(reported_completeness)
            if reported_completeness is not None
            else "partial_known_lower_bound"
            if exception is not None and unfinished
            else "exact"
        )
        self.state["termination_reason"] = (
            raw.get("status") if raw is not None else "unexpected_exception"
        )
        if exception is not None:
            self.state["failure"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "exception_chain": (
                    raw.get("failure_traceback") if raw is not None else None
                )
                or traceback.format_exc(),
            }
        elif raw is not None and raw.get("status") != "completed":
            self.state["failure"] = {
                "type": raw.get("failure_type"),
                "stage": raw.get("failure_stage"),
                "message": raw.get("failure_explanation"),
                "exception_chain": raw.get("failure_traceback"),
            }
        self.persist()
        _hash_sidecar(self.path)
        return dict(self.state)


def new_execution_identity(campaign_identity: str, methods: tuple[str, ...]) -> str:
    return stable_hash(
        "paper-execution-invocation-v1",
        {
            "campaign_identity": campaign_identity,
            "methods": methods,
            "nonce": uuid4().hex,
        },
    )


def append_execution_receipt(root: Path, receipt: Mapping[str, Any]) -> Path:
    receipt_id = str(receipt["execution_receipt_id"])
    stem = sha256(receipt_id.encode("utf-8")).hexdigest()[:16]
    path = root / "executions" / f"receipt-{stem}.json"
    if path.exists() or path.with_suffix(".sha256").exists():
        raise PaperContractError(f"append-only execution receipt already exists: {path}")
    write_pretty_json(path, dict(receipt))
    _hash_sidecar(path)
    receipts = []
    for item in sorted((root / "executions").glob("*.json")):
        sidecar = item.with_suffix(".sha256")
        if (
            not sidecar.is_file()
            or file_hash(item) != sidecar.read_text(encoding="ascii").strip()
        ):
            raise PaperContractError(f"execution receipt hash mismatch: {item}")
        receipts.append(
            {"path": item.relative_to(root).as_posix(), "sha256": file_hash(item)}
        )
    write_pretty_json(
        root / "execution_index.json",
        {
            "schema_version": "mutaco-paper-derived-execution-index-v1",
            "classification": "derived_non_authoritative",
            "receipt_count": len(receipts),
            "receipts": receipts,
        },
    )
    return path


def load_execution_receipts(root: Path) -> tuple[dict[str, Any], ...]:
    directory = root / "executions"
    if not directory.exists():
        return ()
    receipts = []
    json_paths = sorted(directory.glob("*.json"))
    sidecar_paths = sorted(directory.glob("*.sha256"))
    if len(json_paths) != len(sidecar_paths):
        raise PaperContractError("incomplete append-only execution receipt set")
    from .validation import read_json_object

    for path in json_paths:
        sidecar = path.with_suffix(".sha256")
        if not sidecar.is_file() or file_hash(path) != sidecar.read_text(
            encoding="ascii"
        ).strip():
            raise PaperContractError(f"execution receipt hash mismatch: {path}")
        receipt = read_json_object(path)
        if receipt.get("schema_version") != "mutaco-paper-execution-receipt-v1":
            raise PaperContractError(f"unsupported execution receipt: {path}")
        for reference in receipt.get("records", ()):
            record = (root / str(reference["record"])).resolve()
            try:
                record.relative_to(root.resolve())
            except ValueError as exc:
                raise PaperContractError(
                    f"execution receipt record escapes campaign root: {record}"
                ) from exc
            if (
                not record.is_file()
                or file_hash(record) != reference.get("record_sha256")
            ):
                raise PaperContractError(
                    f"execution receipt referenced record hash mismatch: {record}"
                )
            journal_name = reference.get("attempt_journal")
            if journal_name is not None:
                journal = (root / str(journal_name)).resolve()
                if (
                    not journal.is_file()
                    or file_hash(journal)
                    != reference.get("attempt_journal_sha256")
                ):
                    raise PaperContractError(
                        "execution receipt referenced attempt-journal hash mismatch: "
                        f"{journal}"
                    )
        receipts.append(receipt)
    receipts.sort(key=lambda item: str(item.get("started_at", "")))
    return tuple(receipts)


def validate_attempt_journals(
    root: Path, receipts: tuple[Mapping[str, Any], ...] = ()
) -> None:
    attempts = root / "attempts"
    if not attempts.exists():
        return
    from .validation import read_json_object

    referenced = {
        str(item["attempt_journal"])
        for receipt in receipts
        for item in receipt.get("records", ())
        if item.get("attempt_journal") is not None
    }
    for path in sorted(attempts.rglob("*.json")):
        document = read_json_object(path)
        sidecar = path.with_suffix(".sha256")
        if document.get("status") == "running" or not sidecar.is_file():
            raise PaperContractError(f"incomplete paper attempt journal: {path}")
        if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
            raise PaperContractError(f"paper attempt journal hash mismatch: {path}")
        relative = path.relative_to(root).as_posix()
        if relative not in referenced:
            raise PaperContractError(
                f"finalized attempt journal lacks an execution receipt: {path}"
            )


def write_mutaco_evidence(root: Path, raw: Mapping[str, Any]) -> tuple[Path, ...]:
    """Write readable evidence without model tensors or duplicate full results."""

    projection = raw.get("scientific_projection")
    runner = raw.get("runner_provenance")
    if projection is None and isinstance(runner, Mapping):
        projection = {
            key: runner.get(key)
            for key in (
                "status",
                "failure_explanation",
                "scientific_identity",
                "execution_identity",
                "initial_design",
                "transform",
                "standardized_target",
                "observations",
                "fits",
                "proposals",
                "recommendation",
                "logical_accounting",
                "physical_accounting",
            )
        }
    evidence = {
        "schema_version": "mutaco-paper-compact-scientific-evidence-v1",
        "target_index": raw.get("target_index"),
        "theta_star": raw.get("theta_star"),
        "target_response": raw.get("target_response"),
        "target_seed": raw.get("target_seed"),
        "mutaco_root_seed": raw.get("mutaco_root_seed"),
        "scientific_projection": projection,
        "fit_diagnostics": runner.get("fits") if isinstance(runner, Mapping) else None,
        "graph_terminal_recommendation": raw.get("graph_terminal_recommendation"),
        "proposal_effort": raw.get("proposal_effort"),
        "resolved_proposal_controls": raw.get("resolved_proposal_controls"),
        "recommended_theta": raw.get("recommended_theta"),
        "assessment_response": raw.get("assessment_response"),
        "assessment_seed": raw.get("assessment_seed"),
        "errors": {
            key: raw.get(key)
            for key in (
                "target_error_l1",
                "target_error_l2",
                "design_error_l1",
                "design_error_l2",
                "latent_error_l1",
                "latent_error_l2",
            )
        },
        "timings": runner.get("timings") if isinstance(runner, Mapping) else None,
    }
    evidence_path = root / "scientific_evidence.json"
    write_pretty_json(evidence_path, evidence)
    _hash_sidecar(evidence_path)
    observations = ()
    if isinstance(projection, Mapping):
        observations = projection.get("observations") or ()
    csv_path = root / "evaluation_history.csv"
    write_csv(
        csv_path,
        (
            {
                "observation_id": item.get("observation_id"),
                "phase": item.get("phase"),
                "logical_index": item.get("logical_index"),
                "theta": item.get("theta"),
                "response_physical": item.get("response_physical"),
                "evaluation_seed": item.get("evaluation_seed")
                or (item.get("context") or {}).get("evaluation_seed"),
            }
            for item in observations
        ),
        (
            "observation_id",
            "phase",
            "logical_index",
            "theta",
            "response_physical",
            "evaluation_seed",
        ),
    )
    _hash_sidecar(csv_path)
    return evidence_path, csv_path


ProgressCallback = Callable[[Mapping[str, Any]], None]


class ConsoleProgress:
    """Throttled human-readable progress; all state is observational."""

    def __init__(self, interval_seconds: float = 10.0) -> None:
        self.interval_seconds = interval_seconds
        self._last = 0.0

    def __call__(self, event: Mapping[str, Any]) -> None:
        now = perf_counter()
        force = event.get("event") in {
            "method_started",
            "method_completed",
            "method_failed",
            "method_reused",
        }
        if not force and now - self._last < self.interval_seconds:
            return
        self._last = now
        completed = event.get("inference_calls_completed", 0)
        expected = event.get("expected_inference_calls", "?")
        print(
            f"[{event.get('campaign')} | {event.get('method')} | target "
            f"{event.get('target_position')}/{event.get('target_count')}] "
            f"{event.get('phase')}; calls={completed}/{expected}; "
            f"elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s"
        )

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.integrity")
